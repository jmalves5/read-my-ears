import torch
import wandb
import os
import numpy as np
from natsort import natsorted
from TAL_model import TemporalActionLocalizationModel
from plot_qualitative_results import plot_movement, create_ear_annotation_dict

i3d_best_feature_path = "/home/joao/workspace/EquinePainFaceDataset/dataset_balanced/best_i3d_features"

INFER_WINDOW_STEP_PAIRS = [(75, 50), (50, 50), (50, 35), (35, 35), (35, 15), (25, 25), (20, 20)]

FPS = [50]
streams = ["flow"]

num_classes = 1
batch_size = 1
lr = 0.001
epochs = 150
hidden_dim = 256
num_layers = 3
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]

model_path = "lstm_best_epoch_accuracy_hidden_dim_256_num_layers_3_dropout_0.2_stream_flow_lr_0.001_fps_50_window_32_step_16.pth"

model = TemporalActionLocalizationModel(num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))

for fps in FPS:
        for window_size, step_size in INFER_WINDOW_STEP_PAIRS:
                # EquinePainFaceDataset/dataset_balanced/best_i3d_features/50FPS/20/20/S1/i3d
                video_path = f"{i3d_best_feature_path}/{fps}FPS/{window_size}/{step_size}"
                # get list of mp4 files in video_path
                video_list = [f for f in os.listdir(video_path)]
                video_list = natsorted(video_list)
                # natsort video_list
                for video in video_list:
                    video_features_path = f"{video_path}/{video}/i3d"
                    # get list of features in video_features_path
                    video_features_list = [f for f in os.listdir(video_features_path) if "flow" in f]
                    video_features_list = natsorted(video_features_list)
                    mov_predictions = []
                    for video_features_name in video_features_list:
                        # load npy file
                        video_features = np.load(f"{video_features_path}/{video_features_name}") # (T, 1024)
                        # CONVERT TO TENSOR
                        feats = torch.tensor(video_features).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(feats)
                            final_pred = 0
                            for pred in logits.cpu().numpy()[0]:
                                if pred > 0.5:
                                    final_pred = 1
                            mov_predictions.append(final_pred)


                    # mov predictions is a list of 0s and 1s
                    # we will use these to plot the movement
                    # we need to get the ear annotations

                    results_path_base="/home/joao/workspace/i3d-LSTM"
                    annotation_file = "/home/joao/workspace/EquinePainFaceDataset/CleanAnEquinePainFaceDataset/JSONAnnotations/EADs.txt"

                    ear_annotation_dict = create_ear_annotation_dict(annotation_file)

                    os.makedirs(f"{results_path_base}/plots", exist_ok=True)

                    plot_movement(mov_predictions, video+".mp4", ear_annotation_dict, results_path_base, window_size, step_size, fps)