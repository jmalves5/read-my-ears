from ultralytics import YOLO
from tqdm import tqdm 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

from detect_movement import detect_movement
from dataset import EquinePainFaceDatasetMov
from utils.metrics import calculate_metrics
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import configs

import torch
import wandb

import os
import time

# Prefering high recall
SKIP_FRAMES = 5
COLOR_DIFF_THRESHOLD = 2
FLOW_MAG_THRESHOLD = 1
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

    fps = 25 # We only test 25 FPS and we play with the skip frames parameters for different temporal resolutions
    clips_path = f"{configs.DATASET_DIR}/clipped_videos/clip_videos/{fps}FPS"
    test_txt = f"{configs.DATASET_DIR}/clipped_videos/splits/test.txt"
    masked_clips_path = f"{configs.DATASET_DIR}/clipped_videos/face_masked_clips/{fps}FPS"

    # Create output directory if it doesn't exist
    configs.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # create a directory for the test results
    print(f"Creating directory {configs.OUTPUT_DIR}/movement_detection/test")
    
    test_dir = f"{configs.OUTPUT_DIR}/movement_detection/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Directory {test_dir} created")
    else:
        print(f"Directory {test_dir} already exists")

    # init wandb
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="movement-detection-test", name=f"test-{datetime}", mode="disabled")
        
    skip_frames = SKIP_FRAMES
    color_diff_threshold = COLOR_DIFF_THRESHOLD
    flow_mag_threshold = FLOW_MAG_THRESHOLD

    transform = transforms.Resize((1080, 1920))

    # Create the dataset
    dataset = EquinePainFaceDatasetMov(
        clips_folder=clips_path,
        masked_clips_folder=masked_clips_path, 
        annotation_file=test_txt, 
        device=device, 
        fps=fps, 
        model_ear=model_ear, 
        model_face=model_face, 
        transform=transform
    )

    sweep_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_rnn)

    movement_dict_pred = {}
    movement_dict_gt = {}

    # Run Optical flow, build binary movement classification
    for filenames, packed_frames, targets in tqdm(sweep_loader, desc="Processing Data", unit="video"):
        # There are packed so we need to unpack them
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
    precision, recall, f1, accuracy, cm = calculate_metrics(y_gt, y_pred)

    # print metrics in a nice formatted way
    print(f"Movement Detection Metrics:")
    print(f"-------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion matrix:\n{cm}")
            
    # STORE Y_GT, Y_PRED
    #wandb.log({
    #    "y_gt": y_gt,
    #    "y_pred": y_pred,
    #    "precision": precision,
    #    "recall": recall,
    #    "f1": f1,
    #    "accuracy": accuracy,
    #    "confusion_matrix": cm,
    #    "confusion_matrix_plot": fig,
    #})

    #wandb.finish()

    # Save the metrics to a text file
    with open(f"{configs.OUTPUT_DIR}/movement_detection/test/metrics.txt", "w") as f:
        f.write(f"Movement Detection Metrics:\n")
        f.write(f"-------------------------------\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Confusion matrix:\n{cm}\n")

    # Save the confusion matrix figure
    # fig.savefig(f"{configs.OUTPUT_DIR}/movement_detection/test/confusion_matrix.png")

    plt.close("all")

 