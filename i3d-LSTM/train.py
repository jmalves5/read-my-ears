
import math
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import tqdm

from TAL_model import TemporalActionLocalizationModel
from dataset import EquinePainFaceDataset
import configs
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

import wandb
import torch

torch.manual_seed(42)

def train_one_epoch(model, dataset, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_valid = 0  # Tracks total
    # assume batch size is 1
    for feats, labels in loader:
        # Move inputs to device
        feats = feats.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(feats)  # Expected shape: (B, T, 1)

        # Ensure labels match logits shape
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        if labels.shape[1] == 1 and logits.shape[1] > 1:
            labels = labels.expand(-1, logits.shape[1])
        
        labels = labels.unsqueeze(-1)

        # Compute loss
        loss = criterion(logits, labels.float())

        # Track loss
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_valid += batch_size

        # Backward pass
        loss.backward()
        optimizer.step()

    # Compute final loss over valid timesteps (avoid division by zero)
    epoch_loss = running_loss / total_valid if total_valid > 0 else 0.0

    return model, epoch_loss


def validate_one_epoch(model, dataset, loader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0

    total_valid = 0 

    all_preds = []  # Store all predictions
    all_labels = []  # Store all ground truth labels

    with torch.no_grad():  # Disable gradients
        for feats, labels in loader:
            # Move inputs to device
            feats = feats.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(feats)  # Expected shape: (B, T, 1)

            # Ensure labels match logits shape
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            if labels.shape[1] == 1 and logits.shape[1] > 1:
                labels = labels.expand(-1, logits.shape[1])
            
            labels = labels.unsqueeze(-1)

            # Compute loss
            loss = criterion(logits, labels.float())

            # Track loss
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_valid += batch_size

            real_pred = 0

            # If any logit for this sample is > 0.5 set real_pred to 1
            for i in range(logits.shape[1]):
                if logits[0][i] > 0.5:
                    real_pred = 1

            # convert label to int from tensor to array to scalar
            labels = labels.cpu().numpy()
            labels = labels[0][0]
            labels = int(labels)

            # Store predictions and ground truth labels
            all_preds.append(real_pred)
            all_labels.append(labels)

    # Compute final loss over valid timesteps (avoid division by zero)
    epoch_loss = running_loss / total_valid if total_valid > 0 else 0.0

    # Calculate metrics
    precision, recall, f1, accuracy, cm = calculate_metrics(all_preds, all_labels)

    # Log metrics on wandb
    wandb.log({
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_accuracy": accuracy,
        "val_confusion_matrix": cm,
        "cm": cm,
    })

    return epoch_loss, accuracy, all_preds, all_labels



if __name__ == "__main__":
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Weights and Biases sweep configuration
    sweep_config = {
        "method": "grid",  # Perform a grid search
        "parameters": {
            "num_classes": {"values": [1]},
            "batch_size": {"values": [1]},
            "stream": {"values": ["flow"]},
            "lr": {"values": [0.001]},
            "epochs": {"values": [150]},
            "fps": {"values": [50]},
            "window_and_step_size": {"values": [(32, 16)]},
            "hidden_dim": {"values": [256]},
            "num_layers": {"values": [3]},
            "dropout": {"values": [0.2]},
        },
    }

   
    sweep_id = wandb.sweep(sweep_config, project="i3d+LSTM_train_variations")
    #sweep_id = "i3d+LSTM_train_variations/r9o6id5n"


    def sweep_train(config=None):
        with wandb.init(config=config, resume=True):
            config = wandb.config

            features_folder = f"{configs.DATASET_DIR}/clipped_videos/features/augmented_i3d_features_clips/{config.fps}FPS/{config.window_and_step_size[0]}/{config.window_and_step_size[1]}/i3d"

            train_file = f"{configs.DATASET_DIR}/clipped_videos/splits/train.txt"
            train_dataset = EquinePainFaceDataset(
                batch_size=config.batch_size,
                annotation_file=train_file,
                features_folder=features_folder,
                device=device,
                stream=config.stream,
                chunk_size=config.window_and_step_size[0],
                step_size=config.window_and_step_size[1],
                transform=None,
                fps=config.fps
            )

            val_file = f"{configs.DATASET_DIR}/clipped_videos/splits/val.txt"
            val_dataset = EquinePainFaceDataset(
                batch_size=config.batch_size,
                annotation_file=val_file,
                features_folder=features_folder,
                device=device,
                stream=config.stream,
                chunk_size=config.window_and_step_size[0],
                step_size=config.window_and_step_size[1],
                transform=None,
                fps=config.fps
            )

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

            model = TemporalActionLocalizationModel(num_classes=config.num_classes, hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=config.lr)

            # Early Stopping 
            patience = 20  # Number of epochs to wait for improvement before stopping
            best_val_loss = float("inf")
            best_epoch = 0
            no_improve_count = 0

            # Load the trained weights
            script_path = __file__
            # remove filename from path
            script_path = os.path.dirname(script_path)

            # Training loop
            for epoch in tqdm.tqdm(range(config.epochs)):
                model, train_loss = train_one_epoch(model, train_dataset, train_loader, optimizer, criterion, device)
                val_loss, val_accuracy, all_preds, all_labels = validate_one_epoch(model, val_dataset, val_loader, criterion, device)
                
                # write loss in tqdm at start of progress bar
                tqdm.tqdm.write(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}")

                # Log on wandb
                wandb.log({
                   "train_loss": train_loss,
                   "val_loss": val_loss,
                   "val_pred": all_preds,
                   "val_labels": all_labels
                })
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    # Optional: save the best model so far
                    torch.save(
                        model.state_dict(), 
                       f"{script_path}/checkpoints/lstm_best_epoch_accuracy_hidden_dim_{config.hidden_dim}_num_layers_{config.num_layers}_dropout_{config.dropout}_stream_{config.stream}_lr_{config.lr}_fps_{config.fps}_window_{config.window_and_step_size[0]}_step_{config.window_and_step_size[1]}.pth"
                    )
                    best_epoch = epoch
                else:
                    no_improve_count += 1

                # Early stopping check
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch} with no_improvement_count = {no_improve_count} and best_epoch = {best_epoch}")
                    # log model to wandb
                    break
            
            wandb.log_model(path=f"{script_path}/checkpoints/lstm_best_epoch_accuracy_hidden_dim_{config.hidden_dim}_num_layers_{config.num_layers}_dropout_{config.dropout}_stream_{config.stream}_lr_{config.lr}_fps_{config.fps}_window_{config.window_and_step_size[0]}_step_{config.window_and_step_size[1]}.pth")

            print("Training completed")
            wandb.log({
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss
            })
            plt.close('all')
          
    
    # Start the sweep
    wandb.agent(sweep_id, sweep_train)
