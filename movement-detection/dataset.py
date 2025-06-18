import numpy as np
import torch
import cv2
import numpy as np
from locutils.ear_detection import get_ear_frames
import random
import os

from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


def get_list_of_clips_paths(clips_folder, clips_names):
    clips_folder = os.path.abspath(clips_folder)

    # convert to string
    clips_folder = str(clips_folder)
    
    print(f"Getting list of clips paths from {clips_folder}")
    
    class_folders = os.listdir(clips_folder)
    clips_paths = []
    for class_folder in class_folders:
        clips = os.listdir(f"{clips_folder}/{class_folder}")
        for clip in clips:
            if clip.endswith(".mp4") and clip in clips_names:
                clips_paths.append(f"{clips_folder}/{class_folder}/{clip}")

    return clips_paths

def read_txt_to_list(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

class EquinePainFaceDatasetMov(Dataset):
    def __init__(self, clips_folder, masked_clips_folder, annotation_file, device, transform=None, fps=25, model_face=None, model_ear=None):
        self.clips_names = read_txt_to_list(annotation_file)

        self.clips_folder = clips_folder
        self.clips_paths = get_list_of_clips_paths(clips_folder, self.clips_names)

        print("masked clips folder: ", masked_clips_folder)
        self.masked_clips_folder = masked_clips_folder
        self.masked_clips_paths = get_list_of_clips_paths(masked_clips_folder, self.clips_names)

        self.device = device
        self.transform = transform
        self.fps = fps
        self.model_face = model_face
        self.model_ear = model_ear
        
        self.dataset_size = len(self.clips_paths)

        print(f"Len of masked clips paths: {len(self.masked_clips_paths)}")

        self.transform = transform
      

    def __getitem__(self, idx):
        clip_path = self.clips_paths[idx]
        masked_clip_path = self.masked_clips_paths[idx]
        filename = clip_path.split("/")[-1]

        label = 1 if "action" in clip_path else 0

        # print clip path and masked clip path
        print(f"Clip path: {clip_path}")
        print(f"Masked clip path: {masked_clip_path}")

        frames = []
        # read mp4 file and return frames list
        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        masked_frames = []
        
        # read mp4 file and return frames list
        cap = cv2.VideoCapture(masked_clip_path)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            masked_frames.append(frame)
        cap.release()

        print(f"Frames: {len(frames)}")
        print(f"Masked frames: {len(masked_frames)}")

        if len(frames) != len(masked_frames):
            raise ValueError("Frames and masked frames have different lengths: {} vs {}".format(len(frames), len(masked_frames)))

        ear_frames = get_ear_frames(filename, frames, masked_frames, self.model_face, self.model_ear).to('cuda') # stack of tensor (B, C, H, W)
        ear_frames = self.transform(ear_frames)

        return filename, ear_frames, label


    def __len__(self):
        return self.dataset_size