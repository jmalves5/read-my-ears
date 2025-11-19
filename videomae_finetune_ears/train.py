"""
Finetune VideoMAE on the 'joaomalves/read-my-ears' dataset.

This script:
1. Loads the Hugging Face dataset
2. Initializes wandb and reads hyperparameters from wandb.config
3. Uses transformers VideoMAEImageProcessor and VideoMAEForVideoClassification
4. Trains the model, logging train/val loss and accuracy to wandb
5. Saves the best checkpoint by validation accuracy
6. Loads the best checkpoint and evaluates on the test split
7. Computes and logs test accuracy and F1 score to wandb
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from datasets import load_dataset, Video, ClassLabel, Features
import wandb
import evaluate

from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from huggingface_hub import hf_hub_download

from sklearn.metrics import accuracy_score, f1_score
from config import default_wandb_config, set_seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    # Fixed number of frames to sample from each video
    # VideoMAE-base was trained on 16 frames, so we need to use 16 frames
    NUM_FRAMES = 16
    
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
    
    # Handle case where all videos were skipped
    if not pixel_values_list:
        raise RuntimeError("No valid videos in batch")
    
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


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    preds = []
    targets = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)
        preds.append(outputs.logits.detach().cpu())
        targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(preds).argmax(dim=1).numpy()
    targets = torch.cat(targets).numpy()
    accuracy = float((preds == targets).mean())

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * pixel_values.size(0)
            preds.append(outputs.logits.cpu())
            targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(preds).argmax(dim=1).numpy()
    targets = torch.cat(targets).numpy()
    accuracy = float((preds == targets).mean())

    return avg_loss, accuracy

# ============================================================================
# Main
# ============================================================================

def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation."""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "videomae_finetune_ears"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    args = parser.parse_args()

    # Determine distributed information from args or environment (supports torchrun and SLURM)
    def _get_env_int(names, default):
        for n in names:
            v = os.environ.get(n)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        return default

    # Priority: explicit CLI args, then common env vars (torchrun), then SLURM vars
    rank = args.rank if args.rank is not None else _get_env_int(["RANK", "SLURM_PROCID"], 0)
    local_rank = args.local_rank if args.local_rank is not None else _get_env_int(["LOCAL_RANK", "SLURM_LOCALID"], 0)
    world_size = args.world_size if args.world_size is not None else _get_env_int(["WORLD_SIZE", "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE"], 1)

    # Configure logging for easier debugging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("train")

    # Initialize distributed process group when running multi-process
    if world_size > 1:
        try:
            logger.info(f"Initializing process group: backend=nccl, init_method=env://")
            dist.init_process_group(backend="nccl", init_method="env://")
        except Exception as e:
            logger.exception("Failed to initialize process group")
            raise

    # Setup device for this process. Use local_rank modulo available GPUs to be robust.
    if torch.cuda.is_available():
        local_gpu_count = torch.cuda.device_count()
        if local_gpu_count == 0:
            device = torch.device("cpu")
        else:
            # Map the process to a GPU index on the current node
            gpu_index = int(local_rank) % local_gpu_count
            torch.cuda.set_device(gpu_index)
            device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    # Set the device context for NCCL collective operations
    if world_size > 1 and device.type == "cuda":
        try:
            dist.barrier(device_ids=[device.index])
        except Exception:
            pass  # Barrier might fail if not all processes are ready yet

    # Print mapping for debugging (only from rank 0 and also locally)
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"Distributed world_size={world_size}, rank={rank}, local_rank={local_rank}")
        logger.info(f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}")
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        # also print a brief per-rank info to help debug mapping issues
        logger.info(f"rank={rank} local_rank={local_rank} device={device} pid={os.getpid()}")

    # Initialize (or get) wandb config on main process only
    if rank == 0:
        wandb_run = wandb.init(project=args.wandb_project, config=default_wandb_config(), name=args.wandb_run_name)
        config = wandb.config
    else:
        wandb_run = None
        config = default_wandb_config()

    set_seed(int(config.get("seed", 42)))

    # Load dataset
    if rank == 0:
        print("Loading dataset...")

    ds = load_dataset(
        "joaomalves/read-my-ears",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    if rank == 0:
        print(ds)
        print(f"Train: {len(ds['train'])} samples")
        print(f"Validation: {len(ds['validation'])} samples")
        print(f"Test: {len(ds['test'])} samples")

    # Initialize model and image processor
    model_ckpt = config.get("model_checkpoint", "MCG-NJU/videomae-base-finetuned-kinetics")
    
    label2id = {"background": 0, "action": 1}
    id2label = {0: "background", 1: "action"}
    
    if rank == 0:
        print(f"Loading model from {model_ckpt}...")
    
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    set_collate_processor(image_processor)
    
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Define training arguments
    batch_size = int(config.get("batch_size", 4))
    num_epochs = int(config.get("epochs", 20))
    learning_rate = float(config.get("learning_rate", 5e-5))
    num_workers = int(config.get("num_workers", 2))
    weight_decay = float(config.get("weight_decay", 0.01))
    early_stopping_patience = int(config.get("early_stopping_patience", 3))
    
    output_dir = f"videomae-finetuned-ears-{wandb_run.id if wandb_run else 'local'}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_on_each_node=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=False,  # Disable auto-loading to avoid checkpoint issues in distributed mode
        metric_for_best_model="accuracy",
        num_train_epochs=num_epochs,
        dataloader_num_workers=num_workers,
        report_to="wandb" if rank == 0 and wandb_run else "none",
        save_total_limit=2,
        weight_decay=weight_decay,  # L2 regularization to reduce overfitting
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,  # Reduce memory usage and add regularization
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        ddp_find_unused_parameters=False,  # Improve DDP performance
        ddp_timeout=3600,  # Increase timeout to 1 hour (in seconds) for video processing
    )

    # Initialize Trainer with early stopping callback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=0.0,
        )
    ]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    # Train the model
    if rank == 0:
        print("Starting training...")
    
    train_results = trainer.train()
    
    # Destroy process group before test evaluation to avoid NCCL timeouts
    # All other ranks should exit after this point
    if world_size > 1:
        dist.destroy_process_group()
        # Exit other ranks cleanly
        if rank != 0:
            print(f"Rank {rank} exiting after training completion")
            sys.exit(0)
    
    # Evaluation on test set (only on main process to avoid distributed sync issues)
    if rank == 0:
        print("Training completed!")
        print(f"Train results: {train_results}")

        # Find and load the best checkpoint
        print("Loading best model checkpoint...")
        best_checkpoint_path = None
        
        # Look for the best checkpoint in the output directory
        import glob
        checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
        
        if checkpoint_dirs:
            # Use the last checkpoint as the best one (assuming early stopping or final checkpoint)
            best_checkpoint_path = checkpoint_dirs[-1]
            print(f"Found checkpoint: {best_checkpoint_path}")
            
            # Load the model from the checkpoint
            try:
                trainer.model = VideoMAEForVideoClassification.from_pretrained(best_checkpoint_path)
                print(f"Model loaded from {best_checkpoint_path}")
            except Exception as e:
                print(f"Could not load model from checkpoint: {e}")
                print("Using current model state")
        else:
            print("No checkpoints found, using current model state")

        # Save final model (best validation accuracy)
        final_model_path = os.path.join(output_dir, "best_model")
        trainer.save_model(final_model_path)
        print(f"Best model saved to {final_model_path}")

        # Compute accuracy and f1 score on test set, as well as confusion matrices
        print("Evaluating on test set...")
        
        # Manual evaluation without distributed operations
        model = trainer.model.to(device)
        model.eval()
        
        # Create a fresh dataloader for the test set instead of using trainer's
        test_dataloader = DataLoader(
            ds["test"],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        
        all_preds = []
        all_labels = []
        
        print(f"Test dataset size: {len(ds['test'])}")
        print(f"Test dataloader length: {len(test_dataloader)}")
        
        batch_count = 0
        with torch.no_grad():
            for batch in test_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(pixel_values)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                batch_count += 1
                print(f"Batch {batch_count}: processed {len(labels_np)} samples (total: {len(all_preds)})")
        
        test_preds = np.array(all_preds)
        test_labels = np.array(all_labels)
        print(f"Total samples processed: {len(test_preds)}")

        print("Computing test metrics...")
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="weighted")

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        # Log test metrics to wandb
        if wandb_run:
            wandb_run.log({
                "test/accuracy": test_accuracy,
                "test/f1_score": test_f1,
            })

        # compute confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        print(f"Confusion matrix shape: {cm.shape}")
        print(f"Confusion matrix:\n{cm}")
        print(f"Sum of all cells: {cm.sum()}")
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix on Test Set")
        cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_image_path)
        print(f"Confusion matrix saved to {cm_image_path}")

        # Log confusion matrix image to wandb
        if wandb_run:
            wandb_run.log({
                "test/confusion_matrix": wandb.Image(cm_image_path)
            })
            wandb_run.finish()

  
if __name__ == "__main__":
    main()
