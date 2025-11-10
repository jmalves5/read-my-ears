import torch
from torch.utils.data import Dataset
import torchvision.io as tv_io
from transformers import VideoMAEImageProcessor
from pathlib import Path
from typing import Tuple
import numpy as np

# ============================================================================
# Dataset
# ============================================================================

class VideoMAEDataset(Dataset):
    """
    Wraps a Hugging Face dataset split for VideoMAE finetuning.
    
    Loads videos and processes them using VideoMAE image processor.
    Returns pixel_values (preprocessed frames) and labels.
    """

    def __init__(
        self,
        hf_split,
        image_processor: VideoMAEImageProcessor,
        num_frames: int = 16,
    ):
        """
        Args:
            hf_split: A Hugging Face dataset split (e.g., ds["train"])
            image_processor: VideoMAEImageProcessor instance
            num_frames: Number of frames to sample from each video
        """
        self.split = hf_split
        self.image_processor = image_processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.split)

    def _infer_label_from_path(self, path: str) -> int:
        """
        Infer label from filename if not explicitly provided.
        Filenames starting with 'action' -> label 1, else -> label 0.
        """
        if path is None:
            return 0  # Default to 0 if path is missing
        name = Path(path).name.lower()
        return 1 if name.startswith("action") else 0

    def _load_video(self, path: str) -> torch.Tensor:
        """
        Load a video file and sample frames evenly.
        
        Returns:
            torch.Tensor of shape (num_frames, H, W, 3) with values in [0, 255]
        """
        try:
            # Read video: returns (frames, audio, info)
            # frames shape: T x H x W x C (uint8)
            video, _, _ = tv_io.read_video(path, pts_unit="sec")
            if video is None or video.size(0) == 0:
                raise RuntimeError(f"Empty video: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path}: {e}")

        # Sample frames evenly across the video
        T_total = video.shape[0]
        if T_total < 1:
            raise RuntimeError(f"No frames in video {path}")

        # Create evenly-spaced frame indices
        indices = np.linspace(0, T_total - 1, self.num_frames).astype(int)
        sampled_frames = video[indices]  # shape: (num_frames, H, W, 3)

        return sampled_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            Tuple of (pixel_values, label)
            - pixel_values: preprocessed by VideoMAEImageProcessor, shape (num_frames, C, H, W)
            - label: int
        """
        item = self.split[idx]

        # Find video path in common field names
        path = None
        for key in ("path", "video", "file", "video_path"):
            if key in item:
                path = item[key]
                break

        if path is None:
            raise RuntimeError(f"Cannot find video path in item {idx}")

        # Load raw frames: shape (num_frames, H, W, 3), uint8 [0, 255]
        frames = self._load_video(path)

        # Get label
        if "label" in item and item["label"] is not None:
            label = int(item["label"])
        else:
            label = self._infer_label_from_path(path)

        # Process frames with image processor
        # image_processor expects a list of PIL images or numpy arrays
        # We'll convert frames to list of numpy uint8 arrays
        frames_list = [frames[i].numpy() for i in range(frames.shape[0])]

        # Process with VideoMAE processor
        # Returns dict with 'pixel_values' key: shape (1, num_frames, C, H, W)
        processed = self.image_processor(frames_list, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # (num_frames, C, H, W)

        return pixel_values, label


def collate_batch(batch):
    """Collate function for DataLoader."""
    pixel_values_list = []
    labels = []
    for pv, label in batch:
        pixel_values_list.append(pv)
        labels.append(label)
    pixel_values = torch.stack(pixel_values_list)  # (batch_size, num_frames, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"pixel_values": pixel_values}, labels