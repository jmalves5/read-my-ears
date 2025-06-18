import os
import torch

import numpy as np
from torch.utils.data import Dataset

import torch
import numpy as np

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_list_of_feature_paths(feature_folder, features_names):
    # inside feature folder there are action and background folders

    feature_folder = os.path.abspath(feature_folder)
    # convert to string
    feature_folder = str(feature_folder)
    
    print(f"Getting list of clips paths from {feature_folder}")

    class_folders = os.listdir(feature_folder)
    features_paths = []
    for class_folder in class_folders:
        features = os.listdir(f"{feature_folder}/{class_folder}")        
        for feature in features:
            feature_clip_name = feature.split("__")[0]
            feature_clip_mp4 = feature_clip_name + "_.mp4"
            if feature.endswith(".npy") and feature_clip_mp4 in features_names:
                features_paths.append(f"{feature_folder}/{class_folder}/{feature}")

    return features_paths

def read_txt_to_list(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

class EquinePainFaceDataset(Dataset):
    def __init__(self, batch_size, features_folder, annotation_file, device, stream,
                 chunk_size=64, step_size=64, transform=None, fps=25):
        
        self.clips_names = read_txt_to_list(annotation_file)

        self.annotation_file = annotation_file
        self.features_folder = features_folder

        self.features_folder = features_folder
        self.features_paths = get_list_of_feature_paths(features_folder, self.clips_names) # this contains all feature paths, so flow and rgb

        self.device = device
        self.stream = stream
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.fps = fps

    def __getitem__(self, idx):
        features_path = ""
        feats_np = None

        if self.stream == "flow":
            features_path = self.features_paths[idx * 2] # take even features
            with open(features_path, 'rb') as f:
                feats_np = np.load(f)
        elif self.stream == "rgb":
            features_path = self.features_paths[idx * 2 + 1] # take odd features
            with open(features_path, 'rb') as f:
                feats_np = np.load(f)
        elif self.stream == "mixed":
            # we want to read both rg and flow and average the features
            flow_path = self.features_paths[idx * 2] # take even features
            rgb_path = self.features_paths[idx * 2 + 1] # take odd features
            features_path = rgb_path
            with open(rgb_path, 'rb') as f:
                rgb_feats = np.load(f)
            with open(flow_path, 'rb') as f:
                flow_feats = np.load(f)
            feats_np = (rgb_feats + flow_feats) / 2


        # Process to (T, 1024) and (T,1)
        feats = torch.from_numpy(feats_np)  # (T, 1024)
        if feats.shape == torch.Size([1024]):
            feats = feats.unsqueeze(0)  # (1, 1024)
        label = 1 if "action" in features_path else 0
        
        T = feats.shape[0]

        # convert to tensors
        feats = feats.to(self.device)
        label = torch.tensor(label).to(self.device)

        return feats, label # shape is (T, 1024), (1)


    def __len__(self):
        return int(len(self.features_paths)/2) # # divide by 2 because we have two streams
