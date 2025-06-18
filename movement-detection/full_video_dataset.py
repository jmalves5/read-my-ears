import numpy as np
import torch
import cv2
import numpy as np
from locutils.ear_detection import get_ear_frames
import random
import os

from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class EquinePainFaceDatasetQual(Dataset):
    def __init__(self, video_folder, masked_video_folder, window_size, step_size, device, fps=25, model_face=None, model_ear=None):
        self.video_folder = video_folder
        self.masked_video_folder = masked_video_folder
        self.video_paths = [f"{video_folder}/{filename}" for filename in os.listdir(video_folder) if filename.endswith(".mp4")]
        self.masked_video_paths = [f"{masked_video_folder}/{filename}" for filename in os.listdir(masked_video_folder) if filename.endswith(".mp4")]

        self.device = device
        self.fps = fps
        self.model_face = model_face
        self.model_ear = model_ear

        self.window_size = window_size
        self.step_size = step_size

        # each sample corresponds to 20 frames of a video in self.videos
        # the number of samples is the number of windows of 20 frames in all videos
        # if a window has less than 20 frames we pad it with empty frames
        self.n_windows_video_dict = {}
        self.dataset_size = 0
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            # Get number of frames
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Find number of windows
            n_windows_quotient = (n_frames - self.window_size) // self.step_size
            n_windows_remainder = n_frames % self.window_size
            if n_windows_remainder != 0.0:
                n_windows = n_windows_quotient + 1
            else:
                n_windows = n_windows_quotient
            # Store n_windows for each video
            video_name = video_path.split("/")[-1]
            self.n_windows_video_dict[video_name] = n_windows
            self.dataset_size += n_windows        
      

    def __getitem__(self, idx): # each sample is 20 frames of a video to evaluate
        # firs find which video the idx corresponds to
        video_folder = self.video_folder
        masked_video_folder = self.masked_video_folder
        
        sum = 0
        current_video_name = None
        idx_in_video = None 
        for video_path in self.video_paths:
            video_name = video_path.split("/")[-1]
            idx_in_video = idx - sum
            sum += self.n_windows_video_dict[video_name] 
            if idx < sum:
                current_video_name = video_name
                break

        # load window_size frames of normal video 
        frames = []
        cap = cv2.VideoCapture(f"{video_folder}/{current_video_name}")
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        for i in range(self.window_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_in_video * self.step_size + i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        # load window_size frames of masked video
        masked_frames = []
        cap = cv2.VideoCapture(f"{masked_video_folder}/{current_video_name}")
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        for i in range(self.window_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_in_video * self.step_size + i)
            ret, frame = cap.read()
            if ret:
                masked_frames.append(frame)
        cap.release()

        ear_frames = get_ear_frames(current_video_name, frames, masked_frames, self.model_face, self.model_ear).to('cuda') # stack of tensor (B, C, H, W)

        return current_video_name, idx_in_video, ear_frames


    def __len__(self):
        return self.dataset_size