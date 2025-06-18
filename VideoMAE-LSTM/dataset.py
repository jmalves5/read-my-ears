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
            feature_clip_name = feature.split(".npy")[0]
            feature_clip_mp4 = feature_clip_name + ".mp4"
            if feature.endswith(".npy") and feature_clip_mp4 in features_names:
                features_paths.append(f"{feature_folder}/{class_folder}/{feature}")
    
    return features_paths

def read_txt_to_list(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


class EquinePainFaceDataset(Dataset):
    def __init__(self, batch_size, features_folder, annotation_file, device, fps=25):

        self.clips_names = read_txt_to_list(annotation_file)

        self.annotation_file = annotation_file
        self.features_folder = features_folder
        self.features_paths = get_list_of_feature_paths(features_folder, self.clips_names) # this contains all feature paths, so flow and rgb
     
        self.device = device
        self.fps = fps

    def __getitem__(self, idx):
        features_path = self.features_paths[idx]
       
        with open(features_path, 'rb') as f:
            feats_np = np.load(f)

        # Process to (T, 768) and (T,1)
        feats = torch.from_numpy(feats_np)  # (T, 768)
        if feats.shape == torch.Size([768]):
            feats = feats.unsqueeze(0)  # (1, 768)
        
        label = 1 if "action" in features_path else 0        
        
        T = feats.shape[0]

        # convert to tensors
        feats = feats.to(self.device)
        label = torch.tensor(label).to(self.device)

        return feats, label # shape is (T, 768), (1)


    def __len__(self):
        return len(self.features_paths)
