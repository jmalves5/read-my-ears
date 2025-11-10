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
from torch.utils.data import DataLoader
import numpy as np

from datasets import load_dataset, Video, ClassLabel, Features
import wandb
import evaluate

from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from config import default_wandb_config, set_seed


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(examples):
    """
    Collate function for batching video examples.
    Permutes video dimensions to (num_frames, num_channels, height, width).
    """
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


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
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            outputs = model(pixel_values)
            preds.append(outputs.logits.cpu())
            targets.append(labels)

    preds = torch.cat(preds).argmax(dim=1).numpy()
    targets = torch.cat(targets).numpy()

    return preds, targets



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

    features = Features({
        "video": Video(),
        "label": ClassLabel(names=["background", "action"]),
    })

    ds = load_dataset(
        "joaomalves/read-my-ears",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
        features=features,
    )

    # Cast columns to the correct types for each split in the DatasetDict
    for split in ds.keys():
        ds[split] = ds[split].cast_column("video", Video())
        ds[split] = ds[split].cast_column("label", ClassLabel(names=["background", "action"]))

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
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Define training arguments
    batch_size = int(config.get("batch_size", 4))
    num_epochs = int(config.get("epochs", 10))
    learning_rate = float(config.get("learning_rate", 5e-5))
    num_workers = int(config.get("num_workers", 2))
    
    output_dir = f"videomae-finetuned-ears-{wandb_run.id if wandb_run else 'local'}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=num_epochs,
        dataloader_num_workers=num_workers,
        report_to="wandb" if rank == 0 and wandb_run else "none",
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Train the model
    if rank == 0:
        print("Starting training...")
    
    train_results = trainer.train()
    
    if rank == 0:
        print("Training completed!")
        print(f"Train results: {train_results}")

    # Evaluate on test set
    if rank == 0:
        print("\nEvaluating on test set...")
    
    test_results = trainer.evaluate(ds["test"])
    
    if rank == 0:
        print(f"Test results: {test_results}")
        
        # Log test metrics to wandb
        if wandb_run:
            wandb.log({
                "test/accuracy": test_results.get("eval_accuracy", 0),
                "test/loss": test_results.get("eval_loss", 0),
            })

    # Save the final model
    if rank == 0:
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
        
        # Compute F1 score on test set
        print("\nComputing F1 score on test set...")
        test_dataloader = trainer.get_eval_dataloader(ds["test"])
        preds, targets = evaluate_test(model, test_dataloader, device)
        
        test_accuracy = accuracy_score(targets, preds)
        test_f1 = f1_score(targets, preds, average="weighted")
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        if wandb_run:
            wandb.log({
                "test/accuracy_final": test_accuracy,
                "test/f1_score": test_f1,
            })
            wandb.finish()

    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()

  
if __name__ == "__main__":
    main()
