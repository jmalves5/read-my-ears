import cv2
import numpy as np
import argparse
from locutils.flow import compute_flow, variance_of_flow_mag
from locutils.plot import plot_movement, classify_bool_movement
import os
from detect_movement import detect_movement
import torch
import tqdm
from full_video_dataset import EquinePainFaceDatasetQual
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path

import configs

# Prefering high recall
SKIP_FRAMES = 5
FLOW_MAG_THRESHOLD = 1.2
BATCH_SIZE = 1

# Clipping videos to smaller clips
WINDOW_SIZE = 20 # 0.8 seconds in every window
STEP_SIZE = 5 # A window every 0.2 seconds

if __name__ == "__main__":
    masked_video_paths_base = f"{configs.DATASET_DIR}/full_videos/videos/face_masked_videos"
    video_paths_base = f"{configs.DATASET_DIR}/full_videos/videos/original"
    json_annot_file = f"{configs.DATASET_DIR}/full_videos/ear_annot.txt"

    # model directory
    EAR_DETECTION_MODEL_PATH = Path("../horse-face-ear-detection/horse_ear_detection/yolov8n_horse_ear_detection.pt")
    FACE_DETECTION_MODEL_PATH = Path("../horse-face-ear-detection/horse_face_detection/yolov8n_horse_face_detection.pt")

    # Load a pretrained YOLOv8n model
    model_face = YOLO(FACE_DETECTION_MODEL_PATH)
    model_ear = YOLO(EAR_DETECTION_MODEL_PATH)

    # Load the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_face.to(device)
    model_ear.to(device)

    model_face.eval()
    model_ear.eval()


    for fps in [25]:
        for skip_frames in [SKIP_FRAMES]:
            videos_path = f"{video_paths_base}/{fps}FPS"
            masked_videos_path = f"{masked_video_paths_base}/{fps}FPS"
            
            dataset = EquinePainFaceDatasetQual(
                video_folder=videos_path,
                masked_video_folder=masked_videos_path,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE,
                device=torch.device("cuda"),
                fps=fps,
                model_face=model_face,
                model_ear=model_ear
            )

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            movement_output_dict = {} 
            mean_flow_mag_dict = {}

            for video_path in dataset.video_paths:
                video_name = video_path.split("/")[-1]
                movement_output_dict[video_name] = np.array([])
                mean_flow_mag_dict[video_name] = np.array([])

            for current_video_name, idx_in_video, ear_frames in tqdm.tqdm(dataloader):
                # compute values
                if ear_frames.nelement() == 0:
                    print(f"Ear frames is None for {current_video_name}. Please handle it. Continuing")
                    # save to dicts
                    movement_output_dict[current_video_name[0]] = np.append(
                        movement_output_dict[current_video_name[0]], 
                        float(0)
                    )

                    mean_flow_mag_dict[current_video_name[0]] = np.append(
                        mean_flow_mag_dict[current_video_name[0]], 
                        float(0)
                    )
                    continue
                output, mean_flow_mag = detect_movement(ear_frames, skip_frames, flow_mag_threshold=FLOW_MAG_THRESHOLD)
                # save to dicts
                movement_output_dict[current_video_name[0]] = np.append(
                    movement_output_dict[current_video_name[0]], 
                    float(output[0].cpu().numpy())
                )

                mean_flow_mag_dict[current_video_name[0]] = np.append(
                    mean_flow_mag_dict[current_video_name[0]], 
                    float(mean_flow_mag[0].cpu().numpy())
                )

            # For each timestep we should save the movement_output_dict, the color diff and flow mag to plot
            # create folder to save the results
            # create folder if does not exist
            output_folder = f"{configs.OUTPUT_DIR}/movement_detection/qualitative_movement/25FPS"
            # create folder if does not exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # write dict values to txt files using key as filename
            for key in movement_output_dict.keys():
                np.savetxt(f"{output_folder}/{key}_movement_output.txt", movement_output_dict[key])
                np.savetxt(f"{output_folder}/{key}_flow_mag.txt", mean_flow_mag_dict[key])

   