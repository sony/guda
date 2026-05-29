"""
Utility functions for ReTrack-style Unlearning on UnlearnCanvas.
Includes LoRA insertion, time weighting, fixed random seeds, and I/O helpers.
"""

import os
import random
import yaml
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from safetensors.torch import save_file, load_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config file and apply command-line overrides.
    Supports 'inherit' key for config inheritance.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit' in config:
        parent_path = Path(config_path).parent / config['inherit']
        parent_config = load_config(str(parent_path))
        parent_config.update(config)
        config = parent_config
        del config['inherit']
    
    # Apply overrides
    if overrides:
        config.update(overrides)
    
    return config


def get_nested_config(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested config value using dot notation (e.g., 'data.root')."""
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def setup_lora_unet(
    unet: UNet2DConditionModel,
    rank: int = 16,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None
) -> UNet2DConditionModel:
    """
    Apply LoRA to UNet model.
    
    Args:
        unet: UNet model to apply LoRA
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: List of module names to apply LoRA. If None, use default.
    
    Returns:
        UNet with LoRA applied
    """
    if target_modules is None:
        # Default: Apply to attention projection layers
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2"
        ]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    
    unet = get_peft_model(unet, lora_config)
    return unet


def load_lora_weights(unet: UNet2DConditionModel, lora_path: str):
    """Load LoRA weights from safetensors file."""
    state_dict = load_file(lora_path)
    unet.load_state_dict(state_dict, strict=False)
    return unet


def save_lora_weights(unet: UNet2DConditionModel, save_path: str):
    """Save only LoRA weights to safetensors file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract only LoRA parameters
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if 'lora' in name.lower():
            lora_state_dict[name] = param.detach().cpu()
    
    save_file(lora_state_dict, save_path)


def get_time_weights(timesteps: torch.Tensor, mode: str = "cos2") -> torch.Tensor:
    """
    Get time-dependent weights for loss computation.
    
    Args:
        timesteps: Tensor of timesteps [0, 1000)
        mode: Weighting mode ('cos2', 'uniform', 'snr')
    
    Returns:
        Weights tensor of same shape as timesteps
    """
    if mode == "uniform":
        return torch.ones_like(timesteps, dtype=torch.float32)
    
    elif mode == "cos2":
        # cos²(πt/2T) - emphasizes mid to high noise levels
        t_normalized = timesteps.float() / 1000.0
        weights = torch.cos(np.pi * t_normalized / 2) ** 2
        return weights
    
    elif mode == "snr":
        # This public configuration treats SNR weighting as uniform unless a
        # scheduler-specific weighting rule is added by the caller.
        return torch.ones_like(timesteps, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unknown time weighting mode: {mode}")


def create_output_dir(config: Dict[str, Any], forget_style: Optional[str] = None) -> str:
    """Create output directory based on config and optional forget_style."""
    out_dir = get_nested_config(config, 'out.dir', 'outputs')
    
    if forget_style:
        out_dir = os.path.join(out_dir, forget_style)
    
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_config(config: Dict[str, Any], save_path: str):
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_pipeline(
    model_id: str = "stabilityai/stable-diffusion-1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> StableDiffusionPipeline:
    """Load Stable Diffusion pipeline."""
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline = pipeline.to(device)
    return pipeline


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
