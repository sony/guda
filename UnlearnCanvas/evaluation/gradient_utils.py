#!/usr/bin/env python3
"""
Gradient computation utilities for diffusion models.

This module provides utilities for computing per-sample gradients for attribution methods,
following the implementation patterns from AttributeByUnlearning (Wang et al.).

Reference: https://github.com/PeterWang512/AttributeByUnlearning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from diffusers import UNet2DConditionModel, DDPMScheduler


def get_cross_attention_kv_params(model: nn.Module) -> Tuple[List[str], Dict[str, nn.Parameter]]:
    """
    Extract cross-attention Key/Value parameters from a UNet model.
    
    This follows Wang et al.'s parameter selection strategy where only cross-attention
    K/V projections are updated during unlearning to prevent catastrophic forgetting
    while allowing targeted removal of style information.
    
    Reference: AttributeByUnlearning, compute_influence.py lines 15-34
    Pattern: ['attn2'] AND ['to_k', 'to_v']
    
    Args:
        model: UNet2DConditionModel
        
    Returns:
        param_names: List of parameter names matching the pattern
        param_dict: Dictionary mapping names to parameters
    """
    param_names = []
    param_dict = {}
    
    for name, param in model.named_parameters():
        # Check if parameter belongs to cross-attention (attn2) and is K or V projection
        if 'attn2' in name and ('to_k' in name or 'to_v' in name):
            param_names.append(name)
            param_dict[name] = param
            
    return param_names, param_dict


def compute_diffusion_loss(
    model: UNet2DConditionModel,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    timestep: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute DDPM loss for a batch of latents.
    
    This implements the standard DDPM training loss used in Wang et al.:
        L = MSE(noise_pred, noise)
    where noise_pred = UNet(noisy_latents, timestep, text_embeddings)
    
    Reference: AttributeByUnlearning, compute_influence.py lines 163-177
    
    Args:
        model: UNet model
        latents: Clean latents (B, 4, H, W)
        text_embeddings: Text embeddings (B, 77, 768) or (B, num_prompts, 77, 768)
        noise_scheduler: DDPM scheduler for noise addition
        timestep: Timesteps (B,). If None, random sampling
        noise: Noise tensor (B, 4, H, W). If None, random sampling
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        loss: Scalar loss (reduction='mean'/'sum') or per-sample loss (reduction='none')
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    # Sample timestep if not provided
    if timestep is None:
        timestep = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        )
    
    # Sample noise if not provided
    if noise is None:
        noise = torch.randn_like(latents)
    
    # Add noise to latents (scheduler tensors should already be on correct device)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)
    
    # Handle text_embeddings shape: (B, num_prompts, 77, 768) -> use first prompt
    if text_embeddings.ndim == 4:
        text_embeddings = text_embeddings[:, 0]  # (B, 77, 768)
    
    # Predict noise
    model_output = model(noisy_latents, timestep, text_embeddings, return_dict=False)[0]
    
    # Compute MSE loss
    loss = F.mse_loss(model_output, noise, reduction=reduction)
    
    if reduction == 'none':
        # Return per-sample loss: mean over spatial dimensions
        loss = loss.mean(dim=[1, 2, 3])  # (B,)
    
    return loss


def compute_per_sample_gradients(
    model: UNet2DConditionModel,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    param_names: List[str],
    timestep: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample gradients for specified parameters.
    
    This computes ∇_θ L(θ; x_i) for each sample i in the batch independently,
    which is required for Fisher information estimation and influence functions.
    
    Note: This is a simplified version that computes gradients sequentially.
    Wang et al. use torch.func.grad with vmap for efficiency, but we prioritize
    clarity and compatibility here.
    
    Args:
        model: UNet model (must be in training mode)
        latents: Batch of latents (B, 4, H, W)
        text_embeddings: Text embeddings (B, 77, 768) or (B, num_prompts, 77, 768)
        noise_scheduler: DDPM scheduler
        param_names: List of parameter names to compute gradients for
        timestep: Optional pre-sampled timesteps (B,)
        noise: Optional pre-sampled noise (B, 4, H, W)
        
    Returns:
        grads: Dict mapping param_name -> gradients (B, *param_shape)
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    # Sample timestep and noise if not provided
    if timestep is None:
        timestep = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        )
    if noise is None:
        noise = torch.randn_like(latents)
    
    # Initialize gradient storage
    grads = {name: [] for name in param_names}
    
    # Compute per-sample gradients
    for i in range(batch_size):
        # Zero gradients
        model.zero_grad()
        
        # Compute loss for single sample
        loss = compute_diffusion_loss(
            model,
            latents[i:i+1],
            text_embeddings[i:i+1],
            noise_scheduler,
            timestep[i:i+1],
            noise[i:i+1],
            reduction='mean'
        )
        
        # Backward
        loss.backward()
        
        # Collect gradients
        for name, param in model.named_parameters():
            if name in param_names and param.grad is not None:
                grads[name].append(param.grad.detach().clone())
    
    # Stack gradients: (B, *param_shape)
    grads = {name: torch.stack(grad_list, dim=0) for name, grad_list in grads.items()}
    
    return grads


class DiffusionLossWrapper:
    """
    Wrapper for computing diffusion loss with various configurations.
    
    This class provides a consistent interface for computing DDPM loss across
    different gradient-based attribution diagnostics.
    """
    
    def __init__(
        self,
        model: UNet2DConditionModel,
        noise_scheduler: DDPMScheduler,
        device: torch.device
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        
        # Move all scheduler tensors to device at initialization
        # This ensures add_noise() works correctly
        for attr_name in dir(self.noise_scheduler):
            if not attr_name.startswith('_'):
                try:
                    attr = getattr(self.noise_scheduler, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(self.noise_scheduler, attr_name, attr.to(device))
                except (AttributeError, TypeError):
                    continue
        
    def __call__(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute diffusion loss."""
        return compute_diffusion_loss(
            self.model,
            latents.to(self.device),
            text_embeddings.to(self.device),
            self.noise_scheduler,
            timestep.to(self.device) if timestep is not None else None,
            noise.to(self.device) if noise is not None else None,
            reduction=reduction
        )
    
    def compute_loss_batch(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        time_samples: int = 10,
        avg_timesteps: bool = True,
        init_random_seed: int = 0,
        batch_id: int = 0
    ) -> torch.Tensor:
        """
        Compute loss with deterministic noise and uniform timestep sampling.
        
        This implements Wang et al.'s loss computation for baseline and post-unlearning
        evaluation, which uses:
        1. Deterministic noise (seeded by batch_id for reproducibility)
        2. Uniform timestep subsampling (not random)
        
        Reference: AttributeByUnlearning, compute_training_loss.py lines 42-56
        
        Args:
            latents: Batch of latents (B, 4, H, W)
            text_embeddings: Text embeddings (B, 77, 768)
            time_samples: Number of timesteps to sample (default: 10)
            avg_timesteps: Whether to average over timesteps (default: True)
            init_random_seed: Base random seed (default: 0)
            batch_id: Batch ID for seeding (default: 0)
            
        Returns:
            loss: Per-sample loss (B,) if avg_timesteps=True, else (B, time_samples)
        """
        batch_size = latents.shape[0]
        
        # Move inputs to device if not already
        latents = latents.to(self.device)
        text_embeddings = text_embeddings.to(self.device)
        
        # Deterministic noise generation (CRITICAL for reproducibility)
        noise_seed = init_random_seed + batch_id
        generator = torch.Generator(device=self.device)
        generator.manual_seed(noise_seed)
        noise = torch.randn(
            latents.shape,
            device=self.device,
            generator=generator,
            dtype=latents.dtype
        )
        
        # Uniform timestep sampling
        stride = self.noise_scheduler.config.num_train_timesteps // time_samples
        timesteps = torch.arange(0, self.noise_scheduler.config.num_train_timesteps, stride, device=self.device)
        timesteps = timesteps[:time_samples]  # Ensure exact number
        
        # Compute loss for each timestep
        losses = []
        for t in timesteps:
            t_batch = t.repeat(batch_size)
            loss_t = compute_diffusion_loss(
                self.model,
                latents,
                text_embeddings,
                self.noise_scheduler,
                t_batch,
                noise,
                reduction='none'
            )
            losses.append(loss_t)
        
        losses = torch.stack(losses, dim=1)  # (B, time_samples)
        
        if avg_timesteps:
            return losses.mean(dim=1)  # (B,)
        else:
            return losses  # (B, time_samples)
    
    def compute_loss_single_sample(
        self,
        latent: torch.Tensor,
        text_embedding: torch.Tensor,
        global_sample_index: int,
        time_samples: int = 10,
        init_random_seed: int = 0
    ) -> float:
        """
        Compute loss for a single sample using its global index for noise seeding.
        
        This ensures the same noise is used for baseline and post-unlearn computations
        by using the sample's original index in the full dataset, not the batch-local index.
        
        CRITICAL: This fixes the noise seed bug where post-unlearn used different batch_id
        mapping than baseline, resulting in different noise and meaningless influence scores.
        
        Args:
            latent: Single latent (1, 4, 64, 64)
            text_embedding: Text embedding (1, 77, 768)
            global_sample_index: Original index in full dataset (for noise seeding)
            time_samples: Number of timesteps to sample (default: 10)
            init_random_seed: Base random seed (default: 0)
            
        Returns:
            loss: Scalar loss averaged over timesteps
        """
        # Move inputs to device
        latent = latent.to(self.device)
        text_embedding = text_embedding.to(self.device)
        
        # Use global index for noise seed (not batch-level)
        # CRITICAL: Convert to Python int (numpy.int64 causes TypeError)
        noise_seed = int(init_random_seed + global_sample_index)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(noise_seed)
        
        noise = torch.randn(
            latent.shape,
            device=self.device,
            generator=generator,
            dtype=latent.dtype
        )
        
        # Uniform timestep sampling
        stride = self.noise_scheduler.config.num_train_timesteps // time_samples
        timesteps = torch.arange(
            0, self.noise_scheduler.config.num_train_timesteps, stride,
            device=self.device
        )[:time_samples]
        
        # Compute loss for each timestep
        losses = []
        for t in timesteps:
            t_batch = t.unsqueeze(0)  # (1,)
            loss_t = compute_diffusion_loss(
                self.model,
                latent,
                text_embedding,
                self.noise_scheduler,
                t_batch,
                noise,
                reduction='mean'
            )
            losses.append(loss_t.item())
        
        return float(np.mean(losses))
    
    def compute_loss_batch_with_global_indices(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        global_sample_indices: np.ndarray,
        time_samples: int = 10,
        init_random_seed: int = 0
    ) -> np.ndarray:
        """
        Compute loss using global-index-based noise seeding (sample-level).
        
        CRITICAL: Each sample uses its global index directly as noise seed:
            noise_seed = init_random_seed + global_index
        
        This is INDEPENDENT of batch size or processing order, ensuring:
        - Sample at index 800 always uses noise_seed = 0 + 800
        - Sample at index 1500 always uses noise_seed = 0 + 1500
        
        This matches the baseline computation in compute_wang_baseline_selected.py.
        
        Args:
            latents: Batch of latents (B, 4, 64, 64)
            text_embeddings: Text embeddings (B, 77, 768)
            global_sample_indices: Global indices in full dataset (B,)
            time_samples: Number of timesteps to sample
            init_random_seed: Base random seed
            
        Returns:
            losses: Per-sample losses (B,) averaged over timesteps
        """
        current_batch_size = latents.shape[0]
        
        # Move to device
        latents = latents.to(self.device)
        text_embeddings = text_embeddings.to(self.device)
        
        all_losses = []
        
        # Process each sample individually (needed for per-sample noise)
        for i in range(current_batch_size):
            global_idx = int(global_sample_indices[i])
            noise_seed = int(init_random_seed + global_idx)
            
            generator = torch.Generator(device=self.device)
            generator.manual_seed(noise_seed)
            
            latent_single = latents[i:i+1]
            text_emb_single = text_embeddings[i:i+1]
            
            noise = torch.randn(
                latent_single.shape,
                device=self.device,
                generator=generator,
                dtype=latent_single.dtype
            )
            
            # Uniform timestep sampling
            stride = self.noise_scheduler.config.num_train_timesteps // time_samples
            timesteps = torch.arange(
                0, self.noise_scheduler.config.num_train_timesteps, stride,
                device=self.device
            )[:time_samples]
            
            # Compute loss for each timestep
            losses_per_timestep = []
            for t in timesteps:
                t_single = t.unsqueeze(0)
                loss_t = compute_diffusion_loss(
                    self.model,
                    latent_single,
                    text_emb_single,
                    self.noise_scheduler,
                    t_single,
                    noise,
                    reduction='mean'
                )
                losses_per_timestep.append(loss_t.item())
            
            all_losses.append(np.mean(losses_per_timestep))
        
        return np.array(all_losses)


def print_cross_attention_summary(model: UNet2DConditionModel):
    """Print summary of cross-attention K/V parameters."""
    param_names, param_dict = get_cross_attention_kv_params(model)
    
    total_params = sum(p.numel() for p in param_dict.values())
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"Cross-Attention K/V Parameters:")
    print(f"  Total parameters: {total_params:,} ({100 * total_params / total_model_params:.2f}% of model)")
    print(f"  Number of layers: {len(param_names)}")
    print(f"\nExample parameter names:")
    for name in param_names[:5]:
        print(f"  - {name}")
    if len(param_names) > 5:
        print(f"  ... and {len(param_names) - 5} more")
