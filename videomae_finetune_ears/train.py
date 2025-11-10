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
from pathlib import Path
from typing import Dict, Tuple, Optional

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torchvision.io as tv_io
import numpy as np

from datasets import load_dataset
import wandb

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import accuracy_score, f1_score

from dataset import VideoMAEDataset, collate_batch
from config import default_wandb_config, set_seed


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

    for batch, labels in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = labels.to(device)

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
        for batch, labels in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = labels.to(device)

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


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate on test set.
    
    Returns:
        (predictions, targets) both as numpy arrays
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch, labels in loader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)
            preds.append(outputs.logits.cpu())
            targets.append(labels)

    preds = torch.cat(preds).argmax(dim=1).numpy()
    targets = torch.cat(targets).numpy()

    return preds, targets


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "videomae_finetune_ears"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    args = parser.parse_args()

    local_rank = args.local_rank
    rank = args.rank
    world_size = args.world_size

    # Initialize distributed process group if running with multiple processes
    if world_size > 1:
        # rely on env:// (torchrun or srun sets the required env vars)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Setup device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device}")
        print(f"Distributed world_size={world_size}, rank={rank}, local_rank={local_rank}")

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
    ds = load_dataset("joaomalves/read-my-ears")
    if rank == 0:
        print(f"Dataset splits: {ds.keys()}")

    # print dataset format
    if rank == 0:
        print(f"Dataset format: {ds['train'].features}")

    # print video files
    if rank == 0:
        print("Sample video files from training set:")
        for i in range(min(3, len(ds["train"]))):
            item = ds["train"][i]
            path = item.get("path") or item.get("video") or item.get("file")
            print(f"  {i}: {path}")

    # # Extract splits
    # train_split = ds.get("train")
    # val_split = ds.get("validation")
    # test_split = ds.get("test")

    # if train_split is None:
    #     raise RuntimeError("'train' split not found in dataset")
    # if val_split is None:
    #     raise RuntimeError("'validation' split not found in dataset")
    # if test_split is None:
    #     raise RuntimeError("'test' split not found in dataset")

    # if rank == 0:
    #     print(f"Train size: {len(train_split)}, Val size: {len(val_split)}, Test size: {len(test_split)}")

    # # Discover number of classes from training labels
    # all_labels = set()
    # for i in range(min(100, len(train_split))):
    #     item = train_split[i]
    #     if "label" in item and item["label"] is not None:
    #         all_labels.add(int(item["label"]))
    #     else:
    #         # Infer from path
    #         path = item.get("path") or item.get("video") or item.get("file")
    #         if path is not None:
    #             label = 1 if Path(path).name.lower().startswith("action") else 0
    #             all_labels.add(label)
    #         # If path is also None, skip this item (default to label 0 if truly missing)

    # num_classes = max(all_labels) + 1 if all_labels else 2
    # if rank == 0:
    #     print(f"Number of classes: {num_classes}")

    # # Load model and image processor
    # if rank == 0:
    #     print(f"Loading model from {config.get('model_checkpoint')}...")
    # model_ckpt = config.get('model_checkpoint')
    # image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)

    # # Label mappings
    # label2id = {str(i): i for i in range(num_classes)}
    # id2label = {i: str(i) for i in range(num_classes)}

    # model = VideoMAEForVideoClassification.from_pretrained(
    #     model_ckpt,
    #     label2id=label2id,
    #     id2label=id2label,
    #     ignore_mismatched_sizes=True,
    # )
    # model = model.to(device)

    # # Create datasets and loaders
    # print("Creating datasets and loaders...")
    # train_dataset = VideoMAEDataset(
    #     train_split, image_processor, num_frames=int(config.get('num_frames', 16))
    # )
    # val_dataset = VideoMAEDataset(
    #     val_split, image_processor, num_frames=int(config.get('num_frames', 16))
    # )
    # test_dataset = VideoMAEDataset(
    #     test_split, image_processor, num_frames=int(config.get('num_frames', 16))
    # )

    # # Use DistributedSampler when running distributed
    # train_sampler = None
    # val_sampler = None
    # test_sampler = None
    # if world_size > 1:
    #     from torch.utils.data.distributed import DistributedSampler
    #     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    #     val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    #     test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=int(config.get('batch_size', 4)),
    #     shuffle=(train_sampler is None),
    #     num_workers=int(config.get('num_workers', 2)),
    #     collate_fn=collate_batch,
    #     sampler=train_sampler,
    #     pin_memory=torch.cuda.is_available(),
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=int(config.get('batch_size', 4)),
    #     shuffle=False,
    #     num_workers=int(config.get('num_workers', 2)),
    #     collate_fn=collate_batch,
    #     sampler=val_sampler,
    #     pin_memory=torch.cuda.is_available(),
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=int(config.get('batch_size', 4)),
    #     shuffle=False,
    #     num_workers=int(config.get('num_workers', 2)),
    #     collate_fn=collate_batch,
    #     sampler=test_sampler,
    #     pin_memory=torch.cuda.is_available(),
    # )

    # # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 1e-4)))

    # # Training loop
    # checkpoint_dir = Path("checkpoints")
    # checkpoint_dir.mkdir(exist_ok=True)

    # best_val_accuracy = -1.0
    # best_checkpoint_path = None

    # if rank == 0:
    #     print("Starting training...")
    # for epoch in range(1, int(config.get('epochs', 10)) + 1):
    #     if rank == 0:
    #         print(f"\nEpoch {epoch}/{int(config.get('epochs', 10))}")

    #     # If using DistributedSampler, set epoch for shuffling
    #     if train_sampler is not None:
    #         train_sampler.set_epoch(epoch)

    #     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    #     val_loss, val_acc = validate(model, val_loader, device)

    #     if rank == 0:
    #         print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
    #         print(f"  Val loss:   {val_loss:.4f}, Val acc:   {val_acc:.4f}")

    #         # Log to wandb (main process only)
    #         if wandb_run is not None:
    #             wandb.log({
    #                 "epoch": epoch,
    #                 "train/loss": train_loss,
    #                 "train/accuracy": train_acc,
    #                 "val/loss": val_loss,
    #                 "val/accuracy": val_acc,
    #             })

    #     # Save best checkpoint
    #     if rank == 0 and val_acc > best_val_accuracy:
    #         best_val_accuracy = val_acc
    #         best_checkpoint_path = checkpoint_dir / f"best_checkpoint_epoch{epoch}_acc{val_acc:.4f}.pth"
    #         torch.save(
    #             {
    #                 "epoch": epoch,
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #                 "val_accuracy": val_acc,
    #             },
    #             best_checkpoint_path,
    #         )
    #         print(f"  Saved best checkpoint: {best_checkpoint_path}")
    #         if wandb_run is not None:
    #             wandb.save(str(best_checkpoint_path))

    # if best_checkpoint_path is None:
    #     raise RuntimeError("No checkpoint was saved during training!")

    # # Load best checkpoint and evaluate on test
    # # Wait for all processes to finish training and checkpoint saving
    # if world_size > 1 and dist.is_available() and dist.is_initialized():
    #     dist.barrier()

    # if rank == 0:
    #     if best_checkpoint_path is None:
    #         raise RuntimeError("No checkpoint was saved during training!")

    #     print(f"\nLoading best checkpoint: {best_checkpoint_path}")
    #     checkpoint = torch.load(best_checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint["model_state_dict"])

    #     print("Evaluating on test set...")
    #     test_preds, test_targets = evaluate_test(model, test_loader, device)

    #     test_accuracy = accuracy_score(test_targets, test_preds)
    #     # F1 score: use binary for 2 classes, macro otherwise
    #     avg_type = "binary" if num_classes == 2 else "macro"
    #     test_f1 = f1_score(test_targets, test_preds, average=avg_type)

    #     print(f"\nTest Results:")
    #     print(f"  Accuracy: {test_accuracy:.4f}")
    #     print(f"  F1 Score ({avg_type}): {test_f1:.4f}")

    #     # Log to wandb
    #     if wandb_run is not None:
    #         wandb.log({
    #             "test/accuracy": test_accuracy,
    #             "test/f1": test_f1,
    #         })

    #     print("\nTraining and evaluation complete!")
    #     if wandb_run is not None:
    #         wandb.finish()

    # # Cleanup distributed process group
    # if world_size > 1 and dist.is_available() and dist.is_initialized():
    #     dist.destroy_process_group()


if __name__ == "__main__":
    main()
