import math
import os
import configs
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
import wandb

from TAL_model import TemporalActionLocalizationModel
from dataset import EquinePainFaceDataset
from utils.metrics import calculate_metrics

def run_test(model, dataset, loader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0

    total_valid = 0  # Track only valid (non-padded) elements

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


            # Track loss
            batch_size = labels.size(0)
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


    # Calculate metrics
    precision, recall, f1, accuracy, cm = calculate_metrics(all_preds, all_labels)

    # Log metrics on wandb
    wandb.log({
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_accuracy": accuracy,
        "test_confusion_matrix": cm,
        "test_preds": all_preds,
        "test_labels": all_labels   
    })

    return accuracy, all_preds, all_labels


if __name__ == "__main__":
    # Initialize Weights and Biases sweep configuration
    sweep_config = {
        "method": "grid",  # Perform a grid search
        "parameters": {
            "num_classes": {"values": [1]},
            "batch_size": {"values": [1]},
            "lr": {"values": [0.001]},
            "epochs": {"values": [150]},
            "fps": {"values": [50]},
            "sample": {"values": [8]},
            "hidden_dim": {"values": [256]},
            "num_layers": {"values": [2]},
            "dropout": {"values": [0.2]},
        },
    }

    script_path = __file__
    # remove filename from path
    script_path = os.path.dirname(script_path)
    
    sweep_id = wandb.sweep(sweep_config, project="videoMAE+LSTM_test")    

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sweep_test(config=None):
        with wandb.init(config=config):
            config = wandb.config

            features_folder = f"{configs.DATASET_DIR}/clipped_videos/features/augmented_videoMAEfeatures_clips/{config.fps}FPS/sample{config.sample}"

            test_file = f"{configs.DATASET_DIR}/clipped_videos/splits/test.txt"
            test_dataset = EquinePainFaceDataset(
                batch_size=config.batch_size,
                annotation_file=test_file,
                features_folder=features_folder,
                device=device,
                fps=config.fps
            )

            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

            model = TemporalActionLocalizationModel(num_classes=config.num_classes, hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout).to(device)
            criterion = nn.BCELoss()

    
            # Load the trained weights
            model_path = f"{script_path}/checkpoints/lstm_best_epoch_accuracy_lr_{config.lr}_fps_{config.fps}_sample_{config.sample}_hidden_dim_{config.hidden_dim}_num_layers_{config.num_layers}_dropout_{config.dropout}.pth"
            #model_path = "/home/joao/Downloads/lstm_best_epoch_accuracy_lr_0.001_fps_50_sample_8.pth"
            model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))

            accuracy, all_preds, all_labels = run_test(model, test_dataset, test_loader, criterion, device)
    
    # Start the sweep
    wandb.agent(sweep_id, sweep_test)