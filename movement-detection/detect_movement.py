import cv2
import numpy as np
from locutils.flow import compute_flow
import torch

def detect_movement(frames, skip_frames=5, flow_mag_threshold=2.5):

    # frames is a list of tensors (F, C, H, W)
    B = len(frames)   # Batch size
    # output needs to be (B, 1)
    output = torch.zeros(B, 1)
    mean_flow_mag = torch.zeros(B, 1)
    
    mean_magnitudes = []

    # i should go from 0 to B-1
    for i in range(B):
        F, C, H, W = frames[i].shape

        # Process batch
        frames_np = frames[i].cpu().numpy().transpose(0, 2, 3, 1)  # (F, H, W, C)

        # Convert all frames to uint8 once
        frames_np = (frames_np * 255).astype(np.uint8)

        # Initialize first frame
        frame1_np = frames_np[0]

        for j in range(skip_frames, F, skip_frames):  # Skip frames efficiently
            frame2_np = frames_np[j]

            # Compute optical flow
            flow = compute_flow(frame1_np, frame2_np)
            magnitudes, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Check movement conditions
            mean_magnitudes.append(np.mean(magnitudes))

            # Update frame1 for the next iteration
            frame1_np = frame2_np

        mean_flow_mag[i] = torch.tensor(np.mean(mean_magnitudes))
        if np.mean(mean_magnitudes) > flow_mag_threshold:
            output[i]=1
    
    return output, mean_flow_mag