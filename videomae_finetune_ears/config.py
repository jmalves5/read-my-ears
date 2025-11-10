import torch
import numpy as np
from typing import Dict

# ============================================================================
# Configuration and Setup
# ============================================================================

def default_wandb_config() -> Dict:
    """Default hyperparameters."""
    return {
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_frames": 16,
        "warmup_steps": 500,
        "num_workers": 2,
        "seed": 42,
        "model_checkpoint": "MCG-NJU/videomae-base-finetuned-kinetics",
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)