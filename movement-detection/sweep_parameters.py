from ultralytics import YOLO
from tqdm import tqdm 
from dataset import EquinePainFaceDatasetMov
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from detect_movement import detect_movement
from utils.metrics import calculate_metrics

from torchvision import transforms
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch
import wandb

import configs

BATCH_SIZE = 1

def collate_fn_rnn(batch):
    """
    Handles variable-length video sequences and metadata.
    Returns packed sequences along with filenames and labels.
    """
    filenames, frame_sequences, labels = zip(*batch)  # Unpack batch

    # Sort by sequence length (descending)
    sorted_indices = sorted(range(len(frame_sequences)), key=lambda i: frame_sequences[i].shape[0], reverse=True)
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_labels = torch.tensor([labels[i] for i in sorted_indices])  # Convert labels to tensor

    sorted_sequences = [frame_sequences[i] for i in sorted_indices]  # Reorder sequences
    packed_frames = pack_sequence(sorted_sequences, enforce_sorted=True)  # Pack variable-length sequences

    return sorted_filenames, packed_frames, sorted_labels



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pretrained YOLOv8n model
    model_face = YOLO(configs.FACE_DETECTION_MODEL_PATH)
    model_ear = YOLO(configs.EAR_DETECTION_MODEL_PATH)

    # We only test 25 FPS and we play with the skip frames parameters for different temporal resolutions
    fps = 25
    
    clips_path = f"{configs.DATASET_DIR}/clipped_videos/clip_videos/{fps}FPS"
    train_clips = f"{configs.DATASET_DIR}/clipped_videos/splits/train.txt"
    masked_clips_path = f"{configs.DATASET_DIR}/clipped_videos/face_masked_clips/{fps}FPS"

    # Initialize Weights and Biases sweep configuration
    sweep_config = {
        "method": "grid",  # Perform a grid search
        "parameters": {
            "skip_frames": {"values": [2, 5]},
            "flow_mag_threshold": {"values": [0.7, 0.85, 1, 1.2, 1.5, 2]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="movement-detection-sweep-masked-1080p-narrow")

    transform = transforms.Resize((1080, 1920))
    
    def sweep_train(config=None):
        with wandb.init(config=config, mode="disabled"):
            config = wandb.config

            skip_frames = config.get("skip_frames")
            flow_mag_threshold = config.get("flow_mag_threshold")

            # Transforms are defined whitin the dataset class definition
            dataset = EquinePainFaceDatasetMov(
                clips_path,
                masked_clips_path, 
                train_clips, 
                device, 
                fps=fps, 
                model_ear=model_ear, 
                model_face=model_face, 
                transform=transform
            )
            
            sweep_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_rnn)

            movement_dict_pred = {}
            movement_dict_gt = {}

            # what we want to do is to run the model on the batch and collect the predictions
            # then we want to metrics on the predictions
            # Apply face detection to the batch
            # Apply ear detection to the barch
            # Run Optical flow, build binary movement classification
            for filenames, packed_frames, targets in tqdm(sweep_loader, desc="Processing Data", unit="video"):
                # Theae are packed so we need to unpack them
                unpacked_frames, seq_lengths = pad_packed_sequence(packed_frames, batch_first=True)
                # keep only frames from 0 to seq_lengths
                frames = [unpacked_frames[i, :seq_lengths[i]] for i in range(len(seq_lengths))]

                # Analyse ear frames tensor
                output = detect_movement(
                    frames=frames, 
                    skip_frames=int(skip_frames), 
                    flow_mag_threshold=flow_mag_threshold 
                )  # this return a tensor of shape (B, 1)

                for i, filename in enumerate(filenames):
                    movement_dict_pred[filename] = output[i].cpu().numpy()
                    movement_dict_gt[filename] = targets[i].cpu().numpy()
                
                torch.cuda.empty_cache()
            
            y_gt = []
            y_pred = []
            for filename in movement_dict_pred.keys():
                y_gt.append(movement_dict_gt[filename])
                y_pred.append(movement_dict_pred[filename])
            
            # Calculate metrics
            precision, recall, f1, accuracy, cm, fig = calculate_metrics(y_gt, y_pred, "movDet")
            
            # # STORE Y_GT, Y_PRED
            # wandb.log({
            #     "y_gt": y_gt,
            #     "y_pred": y_pred,
            #     "precision": precision,
            #     "recall": recall,
            #     "f1": f1,
            #     "accuracy": accuracy,
            #     "confusion_matrix": cm,
            #     "confusion_matrix_plot": fig,
            # })

            
            # print metrics in a nice formatted way
            print(f"Movement Detection Metrics:")
            print(f"-------------------------------")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Confusion matrix:\n{cm}")


            # Save the confusion matrix figure
            # create folder if it does not exist
            if not os.path.exists(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep"):
                os.makedirs(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep")

            # Print results as csv with header
            if not os.path.exists(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep/results_{skip_frames}_{flow_mag_threshold}.csv"):
                with open(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep/results_{skip_frames}_{flow_mag_threshold}.csv", "w") as f:
                    f.write("skip_frames,flow_mag_threshold,precision,recall,f1,accuracy\n")
            # Append results to csv
            with open(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep/results_{skip_frames}_{flow_mag_threshold}.csv", "a") as f:
                f.write(f"{skip_frames},{flow_mag_threshold},{precision},{recall},{f1},{accuracy}\n")

            # save the figure
            fig.savefig(f"{configs.OUTPUT_DIR}/movement_detection/train_sweep/confusion_matrix_{skip_frames}_{flow_mag_threshold}.png")
            
            # wandb.finish()
            plt.close("all")

    
    wandb.agent(sweep_id, sweep_train)
