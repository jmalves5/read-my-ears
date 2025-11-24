"""
Create a W&B sweep for VideoMAE finetuning.

Run this script once to create the sweep, then use the sweep ID in your training script.
"""

import os
import wandb
from config import wandb_config


def create_sweep():
    """Create a W&B sweep and print the sweep ID."""
    # Login to wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    
    # Create sweep
    sweep_id = wandb.sweep(
        wandb_config(),
        entity="joaomalves",
        project="videomae_finetune_ears"
    )
    
    print(f"Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"\nUse this command to run the sweep:")
    print(f"wandb agent {sweep_id}")
    print(f"\nOr pass the sweep ID to your training script via environment variable:")
    print(f"export WANDB_SWEEP_ID={sweep_id}")
    
    return sweep_id


if __name__ == "__main__":
    create_sweep()
