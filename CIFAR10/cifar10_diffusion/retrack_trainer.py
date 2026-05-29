"""
ReTrack Unlearning Trainer for Diffusion Models.

This module implements the ReTrack (Redirecting the Denoising Trajectory)
method for unlearning specific classes in diffusion models. ReTrack redirects
the denoising trajectory of forget samples towards similar retain samples.

Based on the paper: "ReTrack: Data Unlearning in Diffusion Models through
Redirecting the Denoising Trajectory"
"""

import os
import copy
import math
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
import hashlib
from torch.utils.data import DataLoader, Dataset

from .base_unlearning_trainer import BaseUnlearningTrainer
from .diffusion_utils import DiffusionHelpers, normal_kl
from evaluation.data_manifests import stable_manifest_hash


class IndexedDataset(Dataset):
    """Dataset wrapper that returns indices along with data."""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        return data, label, idx


class ReTrackUnlearningTrainer(BaseUnlearningTrainer):
    """
    ReTrack (Redirecting the Denoising Trajectory) Unlearning Trainer.
    
    ReTrack redirects the denoising trajectory of forget samples towards 
    similar retain samples using k-nearest neighbors based on Euclidean distance in pixel space.
    
    Loss function:
    L = L_retain + λ * L_forget
    
    Where:
    - L_retain: Standard diffusion loss or distillation loss on retain set
    - L_forget: Weighted loss redirecting forget samples to k-nearest retain samples
    """
    
    def __init__(self, args):
        # Validate ReTrack-specific arguments
        if not hasattr(args, 'k_neighbors') or args.k_neighbors is None:
            raise ValueError("k_neighbors must be specified for ReTrack")
        if not hasattr(args, 'lambda_forget') or args.lambda_forget is None:
            raise ValueError("lambda_forget must be specified for ReTrack")
        if not hasattr(args, 'teacher_model_path') or args.teacher_model_path is None:
            raise ValueError("teacher_model_path must be specified for ReTrack")
        
        self.k_neighbors = args.k_neighbors
        self.lambda_forget = args.lambda_forget
        self.teacher_model_path = args.teacher_model_path
        self.retain_loss_type = getattr(args, 'retain_loss_type', 'standard')
        self.kl_cap = getattr(args, 'kl_cap', 1.0)  # KL cap per time-step
        self.use_fast_retrack = getattr(args, 'fast_retrack', False)
        self.verify_fast_retrack = getattr(args, 'verify_fast_retrack', False)
        self.fast_retrack_max_abs_diff = getattr(args, 'fast_retrack_max_abs_diff', 1e-5)
        self.group_manifest_hash = None
        
        # Timestep window for finetuning
        self.min_t = getattr(args, 'min_t', None)
        self.max_t = getattr(args, 'max_t', None)
        
        # Initialize base unlearning trainer
        super().__init__(args)
        
        # Set num_classes for compatibility
        self.num_classes = 100 if args.dataset == "cifar100" else 10
        
        # Setup k-nearest neighbors cache (before model loading)
        self.knn_cache_dir = Path(args.output_dir) / "knn_cache"
        self.knn_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load teacher model AFTER super().__init__ to ensure accelerator is ready
        # This is called after prepare_with_accelerator() so it's safe
        self.teacher_model = self._load_teacher_model()
        
        # Build k-nearest neighbors mapping AFTER teacher model is loaded
        self.knn_mapping = self._build_or_load_knn_mapping()
        
        if self.accelerator.is_main_process:
            print(f"ReTrack configuration:")
            print(f"  k_neighbors: {self.k_neighbors}")
            print(f"  lambda_forget: {self.lambda_forget}")
            print(f"  retain_loss_type: {self.retain_loss_type}")
            print(f"  kl_cap: {self.kl_cap}")
            print(f"  fast_retrack: {self.use_fast_retrack}")
            if self.verify_fast_retrack:
                print(f"  verify_fast_retrack: {self.verify_fast_retrack}")
                print(f"  fast_retrack_max_abs_diff: {self.fast_retrack_max_abs_diff}")
            print(f"  teacher_model: {self.teacher_model_path}")
            print(f"  knn_cache_dir: {self.knn_cache_dir}")
            if self.group_manifest_hash is not None:
                print(f"  group_manifest_hash: {self.group_manifest_hash}")
            if self.min_t is not None or self.max_t is not None:
                min_t_str = str(self.min_t) if self.min_t is not None else "0"
                max_t_str = str(self.max_t) if self.max_t is not None else "T"
                print(f"  timestep_window: [{min_t_str}, {max_t_str}]")

    def setup_dataloaders(self):
        """Setup dataloaders for retention and forgetting data with indices for k-NN lookup."""
        # Call parent method to setup base dataloaders
        super().setup_dataloaders()
        
        # Override with indexed versions for k-NN lookup
        self.train_forget_loader = DataLoader(
            IndexedDataset(self.train_forget_ds), batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        self.train_retain_loader = DataLoader(
            IndexedDataset(self.train_retain_ds), batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        # Update the dummy train_loader for BaseTrainer compatibility
        self.train_loader = self.train_retain_loader

    def _setup_indexed_dataloaders(self):
        """Setup data loaders with index tracking for k-NN lookup."""
        # Create indexed data loaders
        self.train_forget_loader = DataLoader(
            IndexedDataset(self.train_forget_ds), batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        self.train_retain_loader = DataLoader(
            IndexedDataset(self.train_retain_ds), batch_size=self.pair_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        # Update the dummy train_loader for BaseTrainer compatibility
        self.train_loader = self.train_retain_loader

    def prepare_with_accelerator(self):
        """Prepare models and optimizers with accelerator, handling ReTrack-specific loaders."""
        # Save reference to original optimizer before accelerator wrapping
        self.original_optimizer = self.optimizer
        
        self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.ema_model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )
        
        # Also prepare ReTrack-specific loaders
        self.train_retain_loader, self.train_forget_loader = self.accelerator.prepare(
            self.train_retain_loader, self.train_forget_loader
        )

    def _load_teacher_model(self):
        """Load the teacher model for distillation and feature extraction."""
        if self.accelerator.is_main_process:
            print(f"Loading teacher model from: {self.teacher_model_path}")
        
        # Load teacher model with same architecture as main model
        from diffusers import UNet2DModel
        teacher_model = UNet2DModel.from_pretrained(self.teacher_model_path)
        teacher_model = teacher_model.to(self.device).eval()
        
        # Freeze teacher model parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Prepare teacher model with accelerator
        teacher_model = self.accelerator.prepare(teacher_model)
        
        # Verify that teacher and student models are actually different instances
        if self.accelerator.is_main_process:
            print(f"Teacher model ID: {id(teacher_model)}")
            print(f"Student model ID: {id(self.model)}")
            print(f"Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters())}")
            print(f"Student model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        return teacher_model

    def _get_cache_filename(self) -> str:
        """Generate a unique cache filename based on dataset and parameters."""
        manifest = getattr(self, "train_group_manifest", None)
        if manifest is not None and self.group_manifest_hash is None:
            self.group_manifest_hash = stable_manifest_hash(manifest)
        manifest_tag = self.group_manifest_hash or "raw_labels"
        cache_key = f"{self.args.dataset}_{self.unlearn_class}_{self.k_neighbors}_{manifest_tag}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        return f"knn_mapping_{cache_key}_{cache_hash}.pkl"

    def _extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from images using simple pixel-based flattening.
        Uses Euclidean distance in pixel space for k-NN computation.
        """
        # Simple pixel-based features: flatten the images
        with torch.no_grad():
            # Flatten images to vectors for Euclidean distance computation
            features = images.flatten(1).cpu().numpy()  # Shape: (batch_size, height*width*channels)
        
        return features

    def _build_knn_mapping(self) -> Dict[int, List[int]]:
        """
        Build k-nearest neighbors mapping from forget samples to retain samples.
        Uses Euclidean distance in pixel space for k-NN computation.
        Returns a mapping from forget sample indices to k-nearest retain sample indices.
        """
        if self.accelerator.is_main_process:
            print(f"Building k-nearest neighbors mapping (k={self.k_neighbors}) using Euclidean distance...")
        
        # Extract pixel features from retain samples
        retain_features_list = []
        
        # Process retain dataset in batches
        retain_loader = torch.utils.data.DataLoader(
            self.train_retain_ds, batch_size=64, shuffle=False, num_workers=4
        )
        
        for batch_idx, (images, _) in enumerate(tqdm(retain_loader, desc="Processing retain samples")):
            images = images.to(self.device)
            features = self._extract_features(images)
            retain_features_list.append(features)
        
        retain_features = np.concatenate(retain_features_list, axis=0)
        
        # Extract pixel features from forget samples
        forget_features_list = []
        forget_loader = torch.utils.data.DataLoader(
            self.train_forget_ds, batch_size=64, shuffle=False, num_workers=4
        )
        
        for batch_idx, (images, _) in enumerate(tqdm(forget_loader, desc="Processing forget samples")):
            images = images.to(self.device)
            features = self._extract_features(images)
            forget_features_list.append(features)
        
        forget_features = np.concatenate(forget_features_list, axis=0)
        
        # Build k-nearest neighbors using Euclidean distance
        if self.accelerator.is_main_process:
            print("Computing k-nearest neighbors with Euclidean distance...")
        
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto', metric='euclidean')
        nbrs.fit(retain_features)
        
        # Find k-nearest neighbors for each forget sample
        distances, indices = nbrs.kneighbors(forget_features)
        
        # Build mapping with neighbor indices only
        knn_mapping = {}
        for forget_idx in range(len(forget_features)):
            neighbor_indices = [indices[forget_idx, k_idx] for k_idx in range(self.k_neighbors)]
            knn_mapping[forget_idx] = neighbor_indices
        
        if self.accelerator.is_main_process:
            print(f"Built k-NN mapping for {len(knn_mapping)} forget samples using pixel-based Euclidean distance")
        
        return knn_mapping

    def _build_or_load_knn_mapping(self) -> Dict[int, List[int]]:
        """Build or load k-nearest neighbors mapping from cache."""
        cache_file = self.knn_cache_dir / self._get_cache_filename()
        
        if cache_file.exists() and self.accelerator.is_main_process:
            print(f"Loading k-NN mapping from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                knn_mapping = pickle.load(f)
            print(f"Loaded k-NN mapping for {len(knn_mapping)} forget samples")
            return knn_mapping
        
        # Build new mapping
        knn_mapping = self._build_knn_mapping()
        
        # Save to cache
        if self.accelerator.is_main_process:
            print(f"Saving k-NN mapping to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(knn_mapping, f)
        
        return knn_mapping

    def _calculate_retain_loss(self, retain_batch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Calculate retention loss (standard diffusion or distillation)."""
        if self.retain_loss_type == 'distillation':
            # Distillation loss with teacher model
            noise = torch.randn_like(retain_batch)
            noisy_imgs = self.noise_scheduler.add_noise(retain_batch, noise, t)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_pred = self.teacher_model(noisy_imgs, t).sample
            
            # Get student predictions
            student_pred = self.model(noisy_imgs, t).sample
            
            # MSE loss between teacher and student
            retain_loss = F.mse_loss(student_pred, teacher_pred)
            
        else:  # standard
            # Standard diffusion loss
            noise = torch.randn_like(retain_batch)
            noisy_imgs = self.noise_scheduler.add_noise(retain_batch, noise, t)
            model_pred = self.model(noisy_imgs, t).sample
            retain_loss = F.mse_loss(model_pred, noise)
        
        return retain_loss

    def _calculate_forget_loss(
        self,
        forget_batch: torch.Tensor,
        forget_indices: List[int],
        t: torch.Tensor,
        noisy_forget: torch.Tensor | None = None,
        forget_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate forget loss by redirecting forget samples to k-nearest retain samples.
        Uses analytical target and dynamic timestep-dependent weights as per ReTrack paper.
        
        Args:
            forget_batch: Batch of forget images
            forget_indices: Global indices of forget samples in the dataset
            t: Timestep tensor
        """
        total_forget_loss = 0.0
        batch_size = forget_batch.size(0)
        
        if noisy_forget is None or forget_pred is None:
            # Add noise to forget samples
            noise = torch.randn_like(forget_batch)
            noisy_forget = self.noise_scheduler.add_noise(forget_batch, noise, t)

            # Get model predictions for forget samples
            forget_pred = self.model(noisy_forget, t).sample
        
        # Get noise schedule parameters
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(self.device)
        
        # For each forget sample, compute weighted loss with k-nearest retain samples
        for i in range(batch_size):
            forget_idx = forget_indices[i]
            
            if forget_idx not in self.knn_mapping:
                continue
            
            # Get k-nearest retain sample indices
            neighbor_indices = self.knn_mapping[forget_idx]
            
            # Current timestep parameters
            t_i = t[i].item()
            gamma_t = alpha_bar[t_i].sqrt()  # γ_t = √(ᾱ_t)
            sigma_t = (1.0 - alpha_bar[t_i]).sqrt()  # σ_t = √(1 - ᾱ_t)
            
            # Collect retain samples and compute dynamic weights
            retain_samples = []
            weights = []
            
            for retain_idx in neighbor_indices:
                # Get the corresponding retain sample
                retain_sample, _ = self.train_retain_ds[retain_idx]
                retain_sample = retain_sample.to(self.device)
                retain_samples.append(retain_sample)
                
                # Compute timestep-dependent weight: w̃_t(x_t; a_r) ∝ q_t(x_t | a_r)
                # q_t(x_t | a_r) ∝ exp(-||x_t - γ_t * a_r||² / (2σ_t²))
                x_t = noisy_forget[i]
                gamma_a_r = gamma_t * retain_sample
                
                # Compute squared distance in pixel space
                dist_sq = torch.sum((x_t - gamma_a_r) ** 2)
                
                # Gaussian weight (unnormalized)
                weight = torch.exp(-dist_sq / (2 * sigma_t ** 2))
                weights.append(weight)
            
            # Normalize weights
            weights = torch.stack(weights)
            weights = weights / (torch.sum(weights) + 1e-8)  # Avoid division by zero
            
            # Compute weighted analytical target
            weighted_target = torch.zeros_like(forget_pred[i])
            
            for j, retain_sample in enumerate(retain_samples):
                # KL trust-region clipping: decompose correction and apply norm constraint
                x_t = noisy_forget[i]
                a_f = forget_batch[i].to(self.device)
                
                # Decompose: ε (noise term) + δ (correction)
                eps = (x_t - gamma_t * a_f) / sigma_t
                raw_delta = (gamma_t / sigma_t) * (a_f - retain_sample)
                
                # Apply KL constraint: ||δ||₂ ≤ √(2·kl_cap)
                delta_norm = torch.linalg.vector_norm(raw_delta.reshape(-1))
                max_norm_t = torch.tensor(math.sqrt(2.0 * self.kl_cap), device=raw_delta.device, dtype=raw_delta.dtype)
                scale = torch.minimum(torch.tensor(1.0, device=raw_delta.device, dtype=raw_delta.dtype),
                                    max_norm_t / (delta_norm + 1e-8))
                clipped_delta = raw_delta * scale
                
                # Debug assertion (optional)
                if self.accelerator.is_main_process and torch.rand(1).item() < 0.001:  # Sample 0.1% for efficiency
                    assert torch.linalg.vector_norm(clipped_delta.reshape(-1)) <= max_norm_t + 1e-5, \
                        f"Clipping failed: norm={torch.linalg.vector_norm(clipped_delta.reshape(-1)):.6f}, max={max_norm_t:.6f}"
                
                # Use clipped analytical target
                analytical_target = eps + clipped_delta
                
                # Add weighted contribution
                weighted_target += weights[j] * analytical_target
            
            # Calculate MSE loss between model prediction and weighted analytical target
            sample_loss = F.mse_loss(forget_pred[i], weighted_target)
            total_forget_loss += sample_loss
        
        # Average over batch
        if batch_size > 0:
            total_forget_loss = total_forget_loss / batch_size
        
        return total_forget_loss

    def _fetch_retain_samples(self, neighbor_indices: List[int]) -> torch.Tensor:
        """Fetch retain samples in the given order and stack on CPU before moving to device."""
        retain_samples = [self.train_retain_ds[idx][0] for idx in neighbor_indices]
        return torch.stack(retain_samples, dim=0).to(self.device)

    def _calculate_forget_loss_fast(
        self,
        forget_batch: torch.Tensor,
        forget_indices: List[int],
        t: torch.Tensor,
        noisy_forget: torch.Tensor | None = None,
        forget_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Vectorized forget loss with identical math to the reference implementation.
        Preserves dataset access order while reducing inner Python loops.
        """
        total_forget_loss = 0.0
        batch_size = forget_batch.size(0)

        if noisy_forget is None or forget_pred is None:
            noise = torch.randn_like(forget_batch)
            noisy_forget = self.noise_scheduler.add_noise(forget_batch, noise, t)
            forget_pred = self.model(noisy_forget, t).sample

        alpha_bar = self.noise_scheduler.alphas_cumprod.to(self.device)

        for i in range(batch_size):
            forget_idx = forget_indices[i]

            if forget_idx not in self.knn_mapping:
                continue

            neighbor_indices = self.knn_mapping[forget_idx]
            retain_samples = self._fetch_retain_samples(neighbor_indices)

            t_i = t[i].item()
            alpha_bar_t = alpha_bar[t_i]
            gamma_t = alpha_bar_t.sqrt()
            sigma_t = (1.0 - alpha_bar_t).sqrt()

            x_t = noisy_forget[i]
            gamma_a_r = gamma_t * retain_samples

            dist_sq = (x_t - gamma_a_r).flatten(1).pow(2).sum(dim=1)
            weights = torch.exp(-dist_sq / (2 * sigma_t ** 2))
            weights = weights / (torch.sum(weights) + 1e-8)

            a_f = forget_batch[i].to(self.device)
            eps = (x_t - gamma_t * a_f) / sigma_t
            raw_delta = (gamma_t / sigma_t) * (a_f - retain_samples)

            delta_norm = torch.linalg.vector_norm(raw_delta.reshape(raw_delta.size(0), -1), dim=1)
            max_norm_t = torch.tensor(math.sqrt(2.0 * self.kl_cap), device=raw_delta.device, dtype=raw_delta.dtype)
            scale = torch.minimum(
                torch.ones_like(delta_norm),
                max_norm_t / (delta_norm + 1e-8),
            )
            clipped_delta = raw_delta * scale.view(-1, 1, 1, 1)

            if self.accelerator.is_main_process and torch.rand(1).item() < 0.001:
                max_norm = torch.linalg.vector_norm(clipped_delta.reshape(clipped_delta.size(0), -1), dim=1).max()
                assert max_norm <= max_norm_t + 1e-5, (
                    f"Clipping failed: norm={max_norm:.6f}, max={max_norm_t:.6f}"
                )

            analytical_target = eps + clipped_delta
            weighted_target = (weights.view(-1, 1, 1, 1) * analytical_target).sum(dim=0)

            sample_loss = F.mse_loss(forget_pred[i], weighted_target)
            total_forget_loss += sample_loss

        if batch_size > 0:
            total_forget_loss = total_forget_loss / batch_size

        return total_forget_loss

    def _calculate_loss(self, batch_tuple_or_retain=None, forget_batch=None, forget_indices=None, is_validation=False):
        """
        Calculate the combined ReTrack loss: L_retain + λ * L_forget
        
        Args:
            batch_tuple_or_retain: Either tuple of (retain_batch, forget_batch) for training,
                                  or (batch, batch) for validation, or just retain_batch
            forget_batch: Tuple of (images, labels) for forget data (training only)
            forget_indices: Global indices of forget samples in the dataset (training only)
            is_validation: Whether this is validation (affects logging)
            
        Returns:
            Tuple of (total_loss, retain_loss, forget_loss) for compatibility with BaseTrainer
        """
        # Normalize input formats
        retain_batch_local = None
        forget_batch_local = None

        if isinstance(batch_tuple_or_retain, tuple) and len(batch_tuple_or_retain) >= 1:
            retain_batch_local = batch_tuple_or_retain[0]
            if len(batch_tuple_or_retain) >= 2:
                forget_batch_local = batch_tuple_or_retain[1]
        elif batch_tuple_or_retain is not None:
            retain_batch_local = batch_tuple_or_retain

        if forget_batch is not None:
            forget_batch_local = forget_batch

        if retain_batch_local is None and forget_batch_local is None:
            raise ValueError("Either retain_batch or forget_batch must be provided")

        if forget_indices is not None and hasattr(forget_indices, "tolist"):
            forget_indices = forget_indices.tolist()

        device = self.device
        zero_scalar = torch.zeros((), device=device)
        total_loss = zero_scalar
        retain_loss = zero_scalar
        forget_loss = zero_scalar

        if retain_batch_local is not None:
            retain_imgs, _ = retain_batch_local
            retain_imgs = retain_imgs.to(device)
            retain_batch_size = retain_imgs.size(0)

            if retain_batch_size == 0:
                raise ValueError("Retain batch is empty during loss computation")

            # Sample timesteps within the specified window
            min_timestep = self.min_t if self.min_t is not None else 0
            max_timestep = self.max_t if self.max_t is not None else self.noise_scheduler.config.num_train_timesteps
            
            t_retain = torch.randint(
                min_timestep,
                max_timestep,
                (retain_batch_size,),
                device=device,
            ).long()

            retain_loss = self._calculate_retain_loss(retain_imgs, t_retain)
            total_loss = total_loss + retain_loss

        if forget_batch_local is not None and forget_indices is not None:
            forget_imgs, _ = forget_batch_local
            forget_imgs = forget_imgs.to(device)
            forget_batch_size = forget_imgs.size(0)

            if forget_batch_size == 0:
                raise ValueError("Forget batch is empty during loss computation")

            if len(forget_indices) != forget_batch_size:
                forget_indices = forget_indices[:forget_batch_size]

            # Sample timesteps within the specified window
            min_timestep = self.min_t if self.min_t is not None else 0
            max_timestep = self.max_t if self.max_t is not None else self.noise_scheduler.config.num_train_timesteps
            
            t_forget = torch.randint(
                min_timestep,
                max_timestep,
                (forget_batch_size,),
                device=device,
            ).long()

            if self.use_fast_retrack:
                if self.verify_fast_retrack:
                    noise = torch.randn_like(forget_imgs)
                    noisy_forget = self.noise_scheduler.add_noise(forget_imgs, noise, t_forget)
                    forget_pred = self.model(noisy_forget, t_forget).sample
                    fast_loss = self._calculate_forget_loss_fast(
                        forget_imgs,
                        forget_indices,
                        t_forget,
                        noisy_forget=noisy_forget,
                        forget_pred=forget_pred,
                    )
                    slow_loss = self._calculate_forget_loss(
                        forget_imgs,
                        forget_indices,
                        t_forget,
                        noisy_forget=noisy_forget,
                        forget_pred=forget_pred,
                    )
                    diff = torch.abs(fast_loss - slow_loss).item()
                    if diff > self.fast_retrack_max_abs_diff:
                        raise ValueError(
                            f"fast_retrack mismatch: |fast-slow|={diff:.6e} exceeds {self.fast_retrack_max_abs_diff:.6e}"
                        )
                    forget_loss = fast_loss
                else:
                    forget_loss = self._calculate_forget_loss_fast(forget_imgs, forget_indices, t_forget)
            else:
                forget_loss = self._calculate_forget_loss(forget_imgs, forget_indices, t_forget)
            total_loss = total_loss + self.lambda_forget * forget_loss

        if is_validation and self.accelerator.is_main_process:
            print(f"Validation - retain_loss: {retain_loss.item():.6f}, forget_loss: {forget_loss.item():.6f}")

        return total_loss, retain_loss, forget_loss

    def _train_epoch(self, epoch):
        """Training epoch with paired retain/forget batches."""
        self.model.train()
        running_loss = 0.0
        running_retain_loss = 0.0
        running_forget_loss = 0.0
        
        # Create iterators for both loaders
        retain_iter = iter(self.train_retain_loader)
        forget_iter = iter(self.train_forget_loader)
        
        # Use the shorter loader length to avoid issues
        num_batches = min(len(self.train_retain_loader), len(self.train_forget_loader))
        
        pbar = tqdm(range(num_batches), 
                   desc=f"Epoch {epoch+1}/{self.args.epochs}", 
                   disable=not self.accelerator.is_main_process)
        
        for batch_idx in pbar:
            try:
                retain_batch = next(retain_iter)
                forget_batch = next(forget_iter)
            except StopIteration:
                break
            
            # Extract data, labels, and indices from indexed batches
            retain_imgs, retain_labels, retain_indices = retain_batch
            forget_imgs, forget_labels, forget_indices = forget_batch
            
            # Convert to regular batch format for loss calculation
            retain_batch_formatted = (retain_imgs, retain_labels)
            forget_batch_formatted = (forget_imgs, forget_labels)
            
            # Convert indices to list for k-NN lookup
            forget_indices_list = forget_indices.tolist()
            
            self.optimizer.zero_grad()
            
            with self.accelerator.autocast():
                total_loss, retain_loss_value, forget_loss_value = self._calculate_loss(
                    (retain_batch_formatted, forget_batch_formatted),
                    forget_indices=forget_indices_list,
                    is_validation=False
                )
            
            self.accelerator.backward(total_loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # Update EMA
            self._update_ema()
            
            # Update running averages
            running_loss += total_loss.item()
            running_retain_loss += retain_loss_value.item()
            running_forget_loss += forget_loss_value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'retain': f"{running_retain_loss/(batch_idx+1):.4f}",
                'forget': f"{running_forget_loss/(batch_idx+1):.4f}",
                'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
            })
            
            self.global_step += 1
            
            # Log periodically
            if self.global_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                avg_loss = running_loss / (batch_idx + 1)
                avg_retain = running_retain_loss / (batch_idx + 1)
                avg_forget = running_forget_loss / (batch_idx + 1)
                
                print(f"Step {self.global_step}: loss={avg_loss:.4f}, retain={avg_retain:.4f}, forget={avg_forget:.4f}")
                lr = self.lr_scheduler.get_last_lr()[0]
                metrics = self._format_train_metrics(
                    total_loss=avg_loss,
                    learning_rate=lr,
                    extra={
                        "train/retain_loss": avg_retain,
                        "train/forget_loss": avg_forget,
                        "train/lambda_forget": self.lambda_forget,
                        "train/global_step": self.global_step,
                    },
                )
                self._log_metrics(metrics, step=self.global_step)
        
        if self.accelerator.is_main_process:
            avg_loss = running_loss / num_batches
            avg_retain = running_retain_loss / num_batches
            avg_forget = running_forget_loss / num_batches
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Retain: {avg_retain:.4f}, Forget: {avg_forget:.4f}")
            epoch_metrics = {
                "train/epoch_avg_total_loss": avg_loss,
                "train/epoch_avg_retain_loss": avg_retain,
                "train/epoch_avg_forget_loss": avg_forget,
            }
            self._log_metrics(epoch_metrics, step=self.global_step)

    def _validate_epoch(self, epoch):
        """Validation epoch mirroring ESD trainer behavior with k-NN forget loss monitoring."""
        self.model.eval()
        if self.retain_loss_type == 'distillation' and hasattr(self, 'teacher_model') and self.teacher_model is not None:
            self.teacher_model.eval()

        all_total_losses = []
        all_retain_losses = []
        all_forget_losses = []

        if epoch == -1:
            desc = "Initial ReTrack Validation"
            if self.accelerator.is_main_process:
                print("Performing initial ReTrack validation to establish baseline losses...")
        else:
            desc = "Validation"

        with torch.no_grad():
            val_retain_loader = DataLoader(
                self.val_retain_ds, batch_size=self.pair_batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            val_forget_loader = DataLoader(
                IndexedDataset(self.val_forget_ds), batch_size=self.pair_batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

            val_retain_loader, val_forget_loader = self.accelerator.prepare(
                val_retain_loader, val_forget_loader
            )

            retain_iter = iter(val_retain_loader)
            forget_iter = iter(val_forget_loader)
            num_batches = min(len(val_retain_loader), len(val_forget_loader))

            if num_batches == 0:
                raise ValueError("Validation loaders are empty; check dataset splits.")

            pbar = tqdm(range(num_batches), desc=desc, disable=not self.accelerator.is_main_process)
            for _ in pbar:
                try:
                    retain_batch = next(retain_iter)
                    forget_batch = next(forget_iter)
                except StopIteration:
                    break

                retain_imgs, retain_labels = retain_batch[:2]
                forget_imgs, forget_labels, forget_indices = forget_batch
                forget_indices_list = forget_indices.tolist() if hasattr(forget_indices, "tolist") else list(forget_indices)

                retain_batch_formatted = (retain_imgs, retain_labels)
                forget_batch_formatted = (forget_imgs, forget_labels)

                total_loss, retain_loss, forget_loss = self._calculate_loss(
                    (retain_batch_formatted, forget_batch_formatted),
                    forget_indices=forget_indices_list,
                    is_validation=True
                )

                all_total_losses.append(self.accelerator.gather(total_loss).flatten())
                all_retain_losses.append(self.accelerator.gather(retain_loss).flatten())
                all_forget_losses.append(self.accelerator.gather(forget_loss).flatten())

        avg_total = torch.cat(all_total_losses).mean().item()
        avg_retain = torch.cat(all_retain_losses).mean().item()
        avg_forget = torch.cat(all_forget_losses).mean().item()

        if self.accelerator.is_main_process:
            if epoch == -1:
                print(f"[Initial] val_total: {avg_total:.4f}, val_retain: {avg_retain:.4f}, val_forget: {avg_forget:.4f}")
            else:
                print(f"[Epoch {epoch+1:03d}] val_total: {avg_total:.4f}, val_retain: {avg_retain:.4f}, val_forget: {avg_forget:.4f}")

            step_value = self.global_step if epoch >= 0 else 0
            val_logs = self._format_validation_metrics(
                loss=avg_total,
                mse_loss=None,
                vlb_loss=None,
                forget_ratio=self._latest_forget_ratio,
                extra={
                    "val/retain_loss": avg_retain,
                    "val/forget_loss": avg_forget,
                },
            )
            self._log_metrics(val_logs, step=step_value)

        self.accelerator.wait_for_everyone()
        self.model.train()
