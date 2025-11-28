import torch

NUM_FRAMES = 16  # Fixed number of frames to sample from each video. Default VideoMAE uses 16 frames.

# ============================================================================
# Collate Function
# ============================================================================

# Global reference to image processor (will be set during training setup)
_image_processor = None

def set_collate_processor(processor):
    """Set the image processor for the collate function."""
    global _image_processor
    _image_processor = processor

def augment_frames(frames, augmentation_prob=0.3):
    """
    Apply light data augmentation to video frames to reduce overfitting.
    
    Args:
        frames: List of PIL Images
        augmentation_prob: Probability of applying each augmentation
    
    Returns:
        Augmented list of PIL Images
    """
    import random
    from PIL import ImageEnhance
    
    if random.random() > augmentation_prob:
        return frames
    
    augmented_frames = []
    for frame in frames:
        try:
            # Brightness adjustment
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(frame)
                frame = enhancer.enhance(random.uniform(0.85, 1.15))
            
            # Contrast adjustment
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(frame)
                frame = enhancer.enhance(random.uniform(0.85, 1.15))
            
            augmented_frames.append(frame)
        except Exception:
            augmented_frames.append(frame)
    
    return augmented_frames

def collate_fn(examples):
    """
    Collate function for batching video examples.
    Handles video file paths and processes them with the image processor.
    Downloads videos from HuggingFace dataset repository as needed.
    Uses torchcodec for video decoding (non-deprecated alternative to torchvision).
    """
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import numpy as np
    
    # Load and process videos
    pixel_values_list = []
    labels = []
    
    for example in examples:
        video_path = example["video"]
        
        # Download video from HuggingFace dataset repository
        try:
            local_video_path = hf_hub_download(
                repo_id="joaomalves/read-my-ears",
                filename=video_path,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Error downloading video {video_path}: {e}")
            continue
        
        # Read video file using torchcodec (non-deprecated alternative)
        try:
            decoder = VideoDecoder(str(local_video_path))
            
            # Decode all frames from the video by iterating
            frames = []
            frame_idx = 0
            while True:
                try:
                    frame_tensor = decoder[frame_idx]  # Returns tensor of shape (C, H, W)
                    # Convert to numpy array in HWC format for PIL compatibility
                    frame_np = frame_tensor.permute(1, 2, 0).numpy().astype('uint8')  # HWC
                    frames.append(frame_np)
                    frame_idx += 1
                except (IndexError, RuntimeError):
                    # End of video reached
                    break
            
            if len(frames) == 0:
                raise RuntimeError(f"No frames could be decoded from video: {local_video_path}")
            
            video = np.stack(frames)  # Shape: (num_frames, H, W, 3)
        except Exception as e:
            print(f"Error reading video {local_video_path}: {e}")
            continue
        
        # Check if video has frames
        if len(video) == 0:
            print(f"Warning: Video {video_path} has no frames, skipping")
            continue
        
        # Sample a fixed number of frames uniformly from the video
        total_frames = len(video)
        if total_frames < NUM_FRAMES:
            # If video has fewer frames than needed, repeat frames to reach NUM_FRAMES
            indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
        else:
            # Sample uniformly spaced frames
            indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
        
        # sampled_video_frames will be a list of numpy arrays with shape (H, W, 3)
        sampled_video_frames = [video[i] for i in indices]
        
        # Process video with image processor
        if _image_processor is not None:
            # Convert video frames (numpy arrays) to PIL Images
            # The processor expects a list of PIL Images for a single video
            frames = []
            for frame_array in sampled_video_frames:
                # frame_array is already in HWC format
                frame_array_uint8 = frame_array.astype('uint8')
                # Convert to PIL Image
                frame_pil = Image.fromarray(frame_array_uint8)
                frames.append(frame_pil)
            
            # Apply light augmentation to reduce overfitting
            frames = augment_frames(frames, augmentation_prob=0.3)
            
            # Process the video (list of PIL images)
            processed = _image_processor(frames, return_tensors="pt")
            pixel_values_list.append(processed["pixel_values"])
            labels.append(example["label"])
        else:
            # Fallback: convert frames to tensor and normalize
            frame_tensors = [torch.from_numpy(f).float() for f in sampled_video_frames]
            pixel_values = torch.stack(frame_tensors) / 255.0  # Shape: (NUM_FRAMES, H, W, 3)
            # Rearrange to (NUM_FRAMES, 3, H, W) for compatibility
            pixel_values = pixel_values.permute(0, 3, 1, 2)
            pixel_values_list.append(pixel_values)
            labels.append(example["label"])
    
    # Convert labels to integers (background=0, action=1)
    label_map = {"background": 0, "action": 1}
    labels_int = []
    for label in labels:
        if isinstance(label, str):
            if label in label_map:
                labels_int.append(label_map[label])
            else:
                # Try to convert string to int directly
                try:
                    labels_int.append(int(label))
                except ValueError:
                    raise ValueError(f"Unknown label: {label}. Expected 'background' or 'action'.")
        else:
            labels_int.append(int(label))
    
    # Stack all videos into a single batch
    # Each pixel_values has shape (1, num_frames, C, H, W), squeeze to (num_frames, C, H, W)
    # then stack to get (batch_size, num_frames, C, H, W)
    pixel_values_squeezed = [pv.squeeze(0) for pv in pixel_values_list]
    pixel_values = torch.stack(pixel_values_squeezed, dim=0)
    labels_tensor = torch.tensor(labels_int, dtype=torch.long)
    
    return {"pixel_values": pixel_values, "labels": labels_tensor}
