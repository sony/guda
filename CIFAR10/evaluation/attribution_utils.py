"""
Shared utilities for CIFAR-10 attribution analyses.

This module provides:
1. Diffusion loss wrappers
2. Group aggregation functions
3. Data manifest helpers
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


def compute_diffusion_loss_single(
    model,
    image: torch.Tensor,
    timestep: int,
    noise_scheduler,
    noise: Optional[torch.Tensor] = None,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute diffusion loss for a single (image, timestep) pair.
    
    This wraps the standard diffusion training objective:
    L = MSE(ε̂_θ(x_t, t), ε)
    
    Args:
        model: Diffusion model (UNet)
        image: Clean image tensor [C, H, W]
        timestep: Timestep t (scalar or tensor)
        noise_scheduler: DDPMScheduler instance
        noise: Optional pre-sampled noise (for reproducibility)
        return_components: If True, return (loss, noised_latent, noise, pred_noise)
    
    Returns:
        loss: Scalar loss value (or tuple if return_components=True)
    """
    device = next(model.parameters()).device
    
    # Ensure correct shapes
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    image = image.to(device)
    
    # Sample noise if not provided
    if noise is None:
        noise = torch.randn_like(image)
    else:
        noise = noise.to(device)
    
    # Prepare timestep
    if isinstance(timestep, int):
        timestep = torch.tensor([timestep], device=device)
    else:
        timestep = timestep.to(device)
    
    # Add noise according to scheduler
    noised_image = noise_scheduler.add_noise(image, noise, timestep)
    
    # Predict noise
    with torch.enable_grad():
        pred_noise = model(noised_image, timestep).sample
    
    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction='mean')
    
    if return_components:
        return loss, noised_image, noise, pred_noise
    return loss


def compute_diffusion_loss_batch(
    model,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute diffusion loss for a batch of (image, timestep) pairs.
    
    Args:
        model: Diffusion model
        images: Batch of images [B, C, H, W]
        timesteps: Batch of timesteps [B]
        noise_scheduler: DDPMScheduler instance
        noise: Optional pre-sampled noise [B, C, H, W]
    
    Returns:
        loss: Scalar loss averaged over batch
    """
    device = next(model.parameters()).device
    images = images.to(device)
    timesteps = timesteps.to(device)
    
    if noise is None:
        noise = torch.randn_like(images)
    else:
        noise = noise.to(device)
    
    # Add noise
    noised_images = noise_scheduler.add_noise(images, noise, timesteps)
    
    # Predict
    with torch.enable_grad():
        pred_noise = model(noised_images, timesteps).sample
    
    # MSE loss
    loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction='mean')
    return loss


def aggregate_to_groups(
    per_example_scores: np.ndarray,
    group_ids: np.ndarray,
    num_groups: int = 10,
    aggregation: str = 'sum',
    top_k: Optional[int] = None,
    positive_only: bool = False
) -> np.ndarray:
    """
    Aggregate per-training-example attribution scores to per-group scores.
    
    Supports multiple aggregation strategies for systematic evaluation:
    - 'sum': Sum all scores (default, preserves total influence)
    - 'mean': Average scores (normalizes by class size)
    - 'max': Maximum score (identifies top contributor)
    - 'median': Median score (robust to outliers)
    - 'top_k_mean': Average of top-k scores (requires top_k parameter)
    
    Args:
        per_example_scores: Attribution scores [N_train] or [N_query, N_train]
        group_ids: Group ID for each training example [N_train]
        num_groups: Number of groups (10 for CIFAR-10)
        aggregation: Aggregation strategy (see above)
        top_k: Number of top samples to average (only for 'top_k_mean')
        positive_only: If True, only consider positive scores (filter negative influences)
    
    Returns:
        group_scores: [num_groups] or [N_query, num_groups]
    """
    if per_example_scores.ndim == 1:
        # Single query
        group_scores = np.zeros(num_groups, dtype=np.float32)
        for group_id in range(num_groups):
            mask = (group_ids == group_id)
            scores = per_example_scores[mask]
            
            if len(scores) == 0:
                group_scores[group_id] = 0.0
                continue
            
            # Apply positive_only filter
            if positive_only:
                scores = scores[scores > 0]
                if len(scores) == 0:
                    group_scores[group_id] = 0.0
                    continue
            
            # Apply aggregation
            if aggregation == 'sum':
                group_scores[group_id] = scores.sum()
            elif aggregation == 'mean':
                group_scores[group_id] = scores.mean()
            elif aggregation == 'max':
                group_scores[group_id] = scores.max()
            elif aggregation == 'median':
                group_scores[group_id] = np.median(scores)
            elif aggregation == 'top_k_mean':
                if top_k is None:
                    raise ValueError("top_k must be specified for 'top_k_mean' aggregation")
                # Sort and take top-k
                k = min(top_k, len(scores))
                top_k_scores = np.partition(scores, -k)[-k:]
                group_scores[group_id] = top_k_scores.mean()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        return group_scores
    
    elif per_example_scores.ndim == 2:
        # Multiple queries [N_query, N_train]
        n_query = per_example_scores.shape[0]
        group_scores = np.zeros((n_query, num_groups), dtype=np.float32)
        
        for group_id in range(num_groups):
            mask = (group_ids == group_id)
            scores = per_example_scores[:, mask]  # [N_query, N_group_samples]
            
            if scores.shape[1] == 0:
                continue
            
            # Apply positive_only filter per query
            if positive_only:
                # Create a mask for positive values
                pos_mask = scores > 0
                # For each query, compute aggregation only over positive values
                for q_idx in range(n_query):
                    pos_scores = scores[q_idx, pos_mask[q_idx]]
                    if len(pos_scores) == 0:
                        group_scores[q_idx, group_id] = 0.0
                        continue
                    
                    if aggregation == 'sum':
                        group_scores[q_idx, group_id] = pos_scores.sum()
                    elif aggregation == 'mean':
                        group_scores[q_idx, group_id] = pos_scores.mean()
                    elif aggregation == 'max':
                        group_scores[q_idx, group_id] = pos_scores.max()
                    elif aggregation == 'median':
                        group_scores[q_idx, group_id] = np.median(pos_scores)
                    elif aggregation == 'top_k_mean':
                        if top_k is None:
                            raise ValueError("top_k must be specified for 'top_k_mean' aggregation")
                        k = min(top_k, len(pos_scores))
                        top_k_scores = np.partition(pos_scores, -k)[-k:]
                        group_scores[q_idx, group_id] = top_k_scores.mean()
            else:
                # Standard aggregation without positive filter
                if aggregation == 'sum':
                    group_scores[:, group_id] = scores.sum(axis=1)
                elif aggregation == 'mean':
                    group_scores[:, group_id] = scores.mean(axis=1)
                elif aggregation == 'max':
                    group_scores[:, group_id] = scores.max(axis=1)
                elif aggregation == 'median':
                    group_scores[:, group_id] = np.median(scores, axis=1)
                elif aggregation == 'top_k_mean':
                    if top_k is None:
                        raise ValueError("top_k must be specified for 'top_k_mean' aggregation")
                    # For each query, sort and take top-k
                    for q_idx in range(n_query):
                        k = min(top_k, scores.shape[1])
                        top_k_scores = np.partition(scores[q_idx], -k)[-k:]
                        group_scores[q_idx, group_id] = top_k_scores.mean()
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return group_scores
    
    else:
        raise ValueError(f"Invalid per_example_scores shape: {per_example_scores.shape}")


def save_attribution_csv(
    group_scores: np.ndarray,
    query_ids: List[str],
    output_path: Path,
    class_names: Optional[List[str]] = None,
    sort_by_mean: bool = True
):
    """
    Save group attribution scores in standard CSV format.
    
    Output format matches delta_elbo_unlearned_sorted.csv:
    - Rows: query images (image_0000, image_0001, ...)
    - Columns: classes (0, 1, 2, ..., 9)
    - Optional: sorted by mean attribution score
    
    Args:
        group_scores: [N_query, num_groups]
        query_ids: List of query identifiers
        output_path: Output CSV path
        class_names: Optional class names (default: "0", "1", ..., "9")
        sort_by_mean: If True, append sorted version with mean scores
    """
    num_groups = group_scores.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(num_groups)]
    
    # Create DataFrame
    df = pd.DataFrame(
        group_scores,
        index=[f"image_{i:04d}" for i in range(len(query_ids))],
        columns=class_names
    )
    df.index.name = 'image_id'
    
    # Save unsorted
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved attribution scores to {output_path}")
    
    # Save sorted version
    if sort_by_mean:
        mean_scores = group_scores.mean(axis=0)
        sorted_indices = np.argsort(mean_scores)[::-1]  # Descending
        sorted_class_names = [class_names[i] for i in sorted_indices]
        
        df_sorted = df[sorted_class_names].copy()
        sorted_path = output_path.parent / f"{output_path.stem}_sorted{output_path.suffix}"
        df_sorted.to_csv(sorted_path)
        print(f"Saved sorted attribution scores to {sorted_path}")
        
        # Save mean scores for reference
        mean_df = pd.DataFrame({
            'class': sorted_class_names,
            'mean_attribution': mean_scores[sorted_indices]
        })
        mean_path = output_path.parent / f"{output_path.stem}_mean{output_path.suffix}"
        mean_df.to_csv(mean_path, index=False)
        print(f"Saved mean scores to {mean_path}")


def load_group_ids_from_dataset(
    dataset_name: str = 'cifar10',
    split: str = 'train',
    cache_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Load group IDs (class labels) from dataset.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        split: 'train' or 'test'
        cache_dir: Optional cache directory
    
    Returns:
        group_ids: [N] array of group IDs
    """
    import torchvision
    from torchvision import transforms
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=cache_dir or './data',
            train=(split == 'train'),
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=cache_dir or './data',
            train=(split == 'train'),
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Extract labels
    if hasattr(dataset, 'targets'):
        group_ids = np.array(dataset.targets, dtype=np.int64)
    else:
        group_ids = np.array([label for _, label in dataset], dtype=np.int64)
    
    return group_ids


def sample_uniform_timesteps(
    num_samples: int,
    num_timesteps: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample timesteps uniformly for gradient extraction.
    
    Args:
        num_samples: Number of timesteps to sample
        num_timesteps: Total number of diffusion timesteps
        seed: Random seed
    
    Returns:
        timesteps: [num_samples] array of timesteps
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(0, num_timesteps, size=num_samples)
