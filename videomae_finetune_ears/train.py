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
import json

from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from huggingface_hub import hf_hub_download

from sklearn.metrics import accuracy_score, f1_score
from config import broadcast_sweep_config, set_seed, wandb_config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import compute_metrics

from dataset import set_collate_processor, augment_frames, collate_fn


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "videomae_finetune_ears"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", "joaomalves"))
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

    # Initialize distributed process group when running multi-process
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
        except Exception as e:
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

    # Login to wandb (only on main process)
    if rank == 0:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        print("Wandb login successful on main process")
        # Sweep config

    sweep_id = os.environ.get('WANDB_SWEEP_ID')

    def sweep_train(config=None):
        # First, check if this is a stop signal (for non-rank-0 processes)
        if rank != 0:
            # Receive the broadcast to check for stop signal
            _, config_or_none = broadcast_sweep_config(sweep_id=sweep_id, config=config, rank=rank)
            if config_or_none is None:
                # Stop signal received
                return True  # Signal to stop
            config = config_or_none
        
        if rank != 0:
            wandb.init(
                entity="joaomalves",
                project="videomae_finetune_ears",
                config=config,
                mode="disabled"
            )
        else:
            wandb.init(
                entity="joaomalves",
                project="videomae_finetune_ears",
                config=config,
            )

        # On rank 0, wandb.agent() already initialized the run with config
        # We need to get that config and share it
        if rank == 0:
            # If config is passed from agent, use it; otherwise get from wandb.config
            if config is None:
                config = dict(wandb.config)
            print(f"Rank 0 starting run with config: {config}")
            # Broadcast config to all ranks
            _, config = broadcast_sweep_config(sweep_id=sweep_id, config=config, rank=rank)
        
        dist.barrier()
        
        print(f"Config is {wandb.config.as_dict()} and rank is {rank}")

        # Now all ranks have the same config with the sweep parameters
        set_seed(config["seed"])

        # Initialize model and image processor
        model_ckpt = config["model_checkpoint"]
        
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
        batch_size = config["batch_size"]
        num_epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        num_workers = config["num_workers"]
        weight_decay = config["weight_decay"]
        early_stopping_patience = config["early_stopping_patience"]
        warmup_steps = config["warmup_steps"]
        
        # Get wandb run ID for output directory naming
        wandb_run_id = wandb.run.id if rank == 0 and wandb.run else 'local'
        output_dir = f"videomae-finetuned-ears-{wandb_run_id}"

          # Load dataset
        ds = load_dataset(
            "joaomalves/read-my-ears",
            data_files={
                "train": "train.csv",
                "validation": "val.csv",
                "test": "test.csv",
            },
        )

        # Print dataset info
        if rank == 0:
            print(ds)
            print(f"Train: {len(ds['train'])} samples")
            print(f"Validation: {len(ds['validation'])} samples")
            print(f"Test: {len(ds['test'])} samples")

        
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
            report_to="wandb" if rank == 0 else "none",
            save_total_limit=2,
            weight_decay=weight_decay,  # L2 regularization to reduce overfitting
            lr_scheduler_type="cosine",  # Cosine annealing for better convergence
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,  # Reduce memory usage and add regularization
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            ddp_find_unused_parameters=False,  # Improve DDP performance
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
        
        # Synchronize all processes after training
        if world_size > 1:
            dist.barrier()
            # Non-zero ranks should return early to wait for next sweep iteration
            if rank != 0:
                print(f"Rank {rank} exiting after training completion")
                return False  # Signal to continue (not stop)
        
        # Evaluation on test set (only on main process to avoid distributed sync issues)
        if rank == 0:
            print(f"Train results: {train_results}")

            best_checkpoint_path = None
            
            # Find the best checkpoint based on validation accuracy
            import glob
            import json
            checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
            
            if checkpoint_dirs:
                # Find the checkpoint with the best validation accuracy
                best_accuracy = -1
                best_checkpoint = None
                
                for ckpt_dir in checkpoint_dirs:
                    trainer_state_file = os.path.join(ckpt_dir, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        try:
                            with open(trainer_state_file, 'r') as f:
                                trainer_state = json.load(f)
                                # Get the best metric from trainer state
                                if 'best_metric' in trainer_state and trainer_state['best_metric'] is not None:
                                    if trainer_state['best_metric'] > best_accuracy:
                                        best_accuracy = trainer_state['best_metric']
                                        best_checkpoint = ckpt_dir
                        except Exception as e:
                            print(f"Could not read trainer state from {ckpt_dir}: {e}")
                
                # If we found a best checkpoint, use it; otherwise use the last one
                if best_checkpoint:
                    best_checkpoint_path = best_checkpoint
                    print(f"Found best checkpoint with accuracy {best_accuracy:.4f}: {best_checkpoint_path}")
                else:
                    # Fallback to last checkpoint
                    best_checkpoint_path = sorted(checkpoint_dirs)[-1]
                    print(f"Could not determine best checkpoint, using last: {best_checkpoint_path}")
                
                # Load the model from the checkpoint
                try:
                    trainer.model = VideoMAEForVideoClassification.from_pretrained(best_checkpoint_path)
                    print(f"Model loaded from {best_checkpoint_path}")
                except Exception as e:
                    print(f"Could not load model from checkpoint: {e}")
                    print("Using current model state")
            else:
                print("No checkpoints found, using current model state")

            # Save final model
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
            
            test_preds = np.array(all_preds)
            test_labels = np.array(all_labels)
            print(f"Total samples processed: {len(test_preds)}")

            print("Computing test metrics...")
            test_accuracy = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average="weighted")

            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")

            # Log test metrics to wandb
            if wandb.run:
                wandb.log({
                    "test/accuracy": test_accuracy,
                    "test/f1_score": test_f1,
                })

            # compute confusion matrix
            cm = confusion_matrix(test_labels, test_preds)
            print(f"Confusion matrix:\n{cm}")

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix on Test Set")
            cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_image_path)
            print(f"Confusion matrix saved to {cm_image_path}")

            # Log confusion matrix image to wandb
            if wandb.run:
                wandb.log({
                    "test/confusion_matrix": wandb.Image(cm_image_path)
                })
                wandb.finish()
        
        return False  # Signal to continue (not stop)
    
    # Start the sweep - only rank 0 should call wandb.agent
    if rank == 0:
        wandb.agent(sweep_id, function=sweep_train, entity=args.wandb_entity, project=args.wandb_project)
        
        # Signal other ranks to exit by broadcasting a stop signal
        if world_size > 1:
            print("Rank 0: Sending stop signal to all ranks")
            broadcast_sweep_config(sweep_id=None, config=None, rank=rank, stop=True)
    else:
        # Other ranks need to loop and participate in multiple sweep runs
        # They wait for broadcast signals from rank 0
        while True:
            try:
                should_stop = sweep_train()
                if should_stop:
                    print(f"Rank {rank}: Received stop signal, exiting loop")
                    break
            except Exception as e:
                # If broadcast fails or process group is destroyed, exit
                print(f"Rank {rank} exiting due to exception: {e}")
                import traceback
                traceback.print_exc()
                break

    # Clean up
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
  
if __name__ == "__main__":
    main()