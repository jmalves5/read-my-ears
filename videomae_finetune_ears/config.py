import torch
import numpy as np
from typing import Dict
import torch.distributed as dist
import pickle


# ============================================================================
# Configuration and Setup
# ============================================================================
def wandb_config() -> Dict:
    """Default hyperparameters sweep config"""
    return {
        "method": "grid",
        "parameters": {
            "epochs": {"values": [20]},
            "batch_size": {"values": [4, 8]},
            "learning_rate": {"values": [1e-4]},
            "num_frames": {"values": [8, 16]},
            "warmup_steps": {"values": [500]},
            "num_workers": {"values": [2]},
            "seed": {"values": [42]},
            "model_checkpoint": {"values": ["MCG-NJU/videomae-base-finetuned-kinetics"]},
            # Overfitting prevention parameters
            "weight_decay": {"values": [0.01, 0.005]},  # L2 regularization
            "early_stopping_patience": {"values": [3, 5]},  # Stop if val accuracy doesn't improve for 3 epochs
        },
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def broadcast_sweep_config(sweep_id=None, config=None, rank=0, stop=False):
    """
    Broadcast sweep_id and config from rank 0 to all processes.
    Can also broadcast a stop signal to terminate sweep loops.
    
    Args:
        sweep_id: W&B sweep ID (only needed on rank 0)
        config: wandb config dict (only needed on rank 0)
        rank: current process rank
        stop: if True, broadcasts a stop signal instead of config
    
    Returns:
        tuple: (sweep_id, config) or (None, None) if stop signal received
    """
    if rank == 0:
        # Serialize both sweep_id and config, including stop flag
        data = {'sweep_id': sweep_id, 'config': config, 'stop': stop}
        data_bytes = pickle.dumps(data)
        size = len(data_bytes)
    else:
        size = 0
    
    # Broadcast size
    size_tensor = torch.tensor([size], dtype=torch.long, device='cuda')
    dist.broadcast(size_tensor, src=0)
    size = size_tensor.item()
    
    # Broadcast data
    if rank == 0:
        buffer = torch.ByteTensor(list(data_bytes)).cuda()
    else:
        buffer = torch.ByteTensor(size).cuda()
    
    dist.broadcast(buffer, src=0)
    
    # Deserialize
    if rank != 0:
        data_bytes = bytes(buffer.cpu().tolist())
        data = pickle.loads(data_bytes)
        sweep_id = data['sweep_id']
        config = data['config']
        stop = data.get('stop', False)
        
        # Return None, None if stop signal received
        if stop:
            return None, None
    
    return sweep_id, config