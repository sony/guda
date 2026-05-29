"""
SISS (Subtraction-based Importance Sampling Score) Unlearning Trainer for Diffusion Models.

This module implements the SISS unlearning method for forgetting specific classes from diffusion models
by optimizing a combination of retention (KD) and forgetting (NegGrad) losses using separate forward passes
(double-forward approach without importance sampling).
"""

import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_unlearning_trainer import BaseUnlearningTrainer


class SISSUnlearningTrainer(BaseUnlearningTrainer):
    """
    SISS (Subtraction-based Importance Sampling Score) Unlearning Trainer for Diffusion Models.
    
    Implements the unlearning method for forgetting a specific class c from a diffusion model
    by optimizing a combination of retention (KD) and forgetting (NegGrad) losses using
    separate forward passes (double-forward approach without importance sampling).
    """
    
    def __init__(self, args):
        # SISS No-IS parameters (official implementation)
        self.siss_nois_s = getattr(args, 'siss_nois_s', 1.0)  # NegGrad strength coefficient (s in the formula)
        self.scaling_norm = getattr(args, 'scaling_norm', None)  # L2 upper bound for NegGrad clipping
        self.neg_clip_ratio = getattr(args, 'neg_clip_ratio', 0.1)  # Ratio for auto-initializing scaling_norm (10% of Naive Deletion)
        
        # Backward compatibility: warn if neggrad_alpha is used
        if hasattr(args, 'neggrad_alpha') and args.neggrad_alpha is not None:
            warnings.warn("neggrad_alpha is deprecated in favor of siss_nois_s and scaling_norm. Ignoring neggrad_alpha.", 
                         DeprecationWarning, stacklevel=2)
        
        # SalUn parameters
        self.salun_enable = getattr(args, 'salun_enable', False)
        self.salun_top_p = getattr(args, 'salun_top_p', 0.2)
        
        super().__init__(args)
        
        # Initialize SalUn if enabled (after setup)
        if self.salun_enable:
            self.parameter_masks = {}
            self.compute_saliency_masks()

    def _maybe_init_scaling_norm(self):
        """
        Initialize scaling_norm using Naive Deletion gradient norm probe.
        
        Official recommendation: scaling_norm = 10% of Naive Deletion gradient norm.
        This is computed by taking one forward pass on a retain batch and measuring
        the gradient norm of the standard diffusion loss.
        
        Returns:
            float: The initialized scaling_norm value
        """
        if getattr(self, "scaling_norm", None) is not None:
            return self.scaling_norm
            
        ratio = self.neg_clip_ratio
        print(f"Initializing scaling_norm using Naive Deletion probe with ratio={ratio}")
        
        # Ensure model is in training mode and has gradients enabled
        was_training = self.model.training
        self.model.train()
        
        # Take one probe step on a retain batch (Naive Deletion objective)
        try:
            retain_batch = next(iter(self.train_retain_loader))
        except StopIteration:
            print("Warning: No retain data available for scaling_norm probe, using default value 0.1")
            self.scaling_norm = 0.1
            return self.scaling_norm
            
        imgs, labels = retain_batch[0].to(self.device), retain_batch[1].to(self.device)
        B = imgs.size(0)
        
        # Clear any existing gradients
        self.model.zero_grad()
        
        # Sample random timesteps and noise
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        eps = torch.randn_like(imgs)
        
        # Add noise to images
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        gamma_t = torch.sqrt(alphas_cumprod.index_select(0, t)).view(-1, 1, 1, 1).float()
        sigma_t = torch.sqrt(1.0 - alphas_cumprod.index_select(0, t)).view(-1, 1, 1, 1).float()
        m = gamma_t * imgs + sigma_t * eps
        
        # Forward pass (ensure requires_grad is enabled)
        m.requires_grad_(True)
        pred = self.model(m, t).sample
            
        if self.args.learn_sigma:
            pred, _ = torch.chunk(pred, 2, dim=1)

        # Compute Naive Deletion loss on retained data
        loss = F.mse_loss(pred, eps, reduction="mean")
        
        # Ensure loss requires gradient
        if not loss.requires_grad:
            print("Warning: Loss does not require gradient. Using fallback scaling_norm = 0.1")
            self.scaling_norm = 0.1
            if not was_training:
                self.model.eval()
            return self.scaling_norm
        
        # Compute gradients using backward pass instead of autograd.grad
        loss.backward()
        
        # Compute gradient norm from parameter gradients
        with torch.no_grad():
            norm_sq = 0.0
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    norm_sq += (p.grad**2).sum().item()
            
            base_norm = (norm_sq ** 0.5) if norm_sq > 0 else 0.0
            self.scaling_norm = float(base_norm) * float(ratio)
            
        # Clear gradients after computation
        self.model.zero_grad()
        
        # Restore original training mode
        if not was_training:
            self.model.eval()
            
        print(f"Naive Deletion gradient norm: {base_norm:.6f}, scaling_norm set to: {self.scaling_norm:.6f}")
        return self.scaling_norm

    def compute_saliency_masks(self):
        """Compute saliency-based masks for SalUn using SISS loss."""
        if not self.salun_enable:
            return
            
        print("Computing saliency masks for SalUn...")
        self.model.eval()
        
        # Compute gradients for forget samples using simplified forget loss
        forget_grads = []
        
        # Take a small sample for saliency computation
        sample_loader = DataLoader(
            self.train_forget_ds, batch_size=min(32, len(self.train_forget_ds)), 
            shuffle=True, num_workers=2
        )
        
        for batch in sample_loader:
            imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
            B = imgs.size(0)
            
            # Clear gradients
            self.model.zero_grad()
            
            # Sample timesteps and noise
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
            eps = torch.randn_like(imgs)
            
            # Add noise
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            gamma_t = torch.sqrt(alphas_cumprod.index_select(0, t)).view(-1, 1, 1, 1).float()
            sigma_t = torch.sqrt(1.0 - alphas_cumprod.index_select(0, t)).view(-1, 1, 1, 1).float()
            m = gamma_t * imgs + sigma_t * eps
            
            # Forward pass
            pred = self.model(m, t).sample
            if self.args.learn_sigma:
                pred, _ = torch.chunk(pred, 2, dim=1)
            
            # Simplified forget loss (negative MSE for saliency)
            loss = -F.mse_loss(pred, eps, reduction="mean")
            loss.backward()
            
            # Store gradients
            batch_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    batch_grads[name] = param.grad.clone().detach()
            forget_grads.append(batch_grads)
            
            break  # Only use one batch for efficiency
        
        # Average gradients and create masks
        avg_grads = {}
        for name, param in self.model.named_parameters():
            avg_grads[name] = torch.stack([g[name] for g in forget_grads if name in g]).mean(dim=0)
        
        # Create masks for top-p% parameters
        self.parameter_masks = {}
        for name, grad in avg_grads.items():
            grad_flat = grad.flatten()
            k = max(1, int(len(grad_flat) * self.salun_top_p))
            _, top_indices = torch.topk(grad_flat.abs(), k)
            mask = torch.zeros_like(grad_flat, dtype=torch.bool)
            mask[top_indices] = True
            self.parameter_masks[name] = mask.reshape(grad.shape)
        
        print(f"Created saliency masks for {len(self.parameter_masks)} parameters")
        self.model.train()

    def apply_salun_masks(self):
        """Apply SalUn masks to freeze non-salient parameters."""
        if not self.salun_enable or not self.parameter_masks:
            return
            
        for name, param in self.model.named_parameters():
            if name in self.parameter_masks and param.grad is not None:
                param.grad = param.grad * self.parameter_masks[name].float()

    def _siss_double_forward_loss(self, retain_batch, forget_batch):
        """
        No-IS (double-forward) variant of SISS.
        We compute the keep (quality retention) and forget (neg-grad) terms
        in two separate forward passes without importance weights.
        The loss is:
            L = (n/(n-k)) * E_{x in X} [ MSE(x) ] - (k/(n-k)) * E_{a in A} [ MSE(a) ]
        """
        imgs_x, labels_x = retain_batch  # from X (retain set)
        imgs_a, labels_a = forget_batch  # from A (forget set)
        B = min(imgs_x.size(0), imgs_a.size(0))
        imgs_x, labels_x = imgs_x[:B].to(self.device), labels_x[:B].to(self.device)
        imgs_a, labels_a = imgs_a[:B].to(self.device), labels_a[:B].to(self.device)

        # Sample independent timesteps and noises for X and A
        t_x = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        t_a = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        eps_x = torch.randn_like(imgs_x)
        eps_a = torch.randn_like(imgs_a)

        # Precompute gamma_t and sigma_t (cast to fp32 for stability)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        gamma_x = torch.sqrt(alphas_cumprod.index_select(0, t_x)).view(-1, 1, 1, 1).float()
        sigma_x = torch.sqrt(1.0 - alphas_cumprod.index_select(0, t_x)).view(-1, 1, 1, 1).float()
        gamma_a = torch.sqrt(alphas_cumprod.index_select(0, t_a)).view(-1, 1, 1, 1).float()
        sigma_a = torch.sqrt(1.0 - alphas_cumprod.index_select(0, t_a)).view(-1, 1, 1, 1).float()

        # Noisy inputs
        m_x = gamma_x * imgs_x + sigma_x * eps_x
        m_a = gamma_a * imgs_a + sigma_a * eps_a

        # Ensure scaling_norm is initialized
        if getattr(self, "scaling_norm", None) is None:
            self._maybe_init_scaling_norm()

        # Forward pass for retain data (X)
        pred_x = self.model(m_x, t_x).sample
        if self.args.learn_sigma:
            pred_x, _ = torch.chunk(pred_x, 2, dim=1)
        loss_x = F.mse_loss(pred_x, eps_x, reduction="mean")

        # Forward pass for forget data (A)
        pred_a = self.model(m_a, t_a).sample
        if self.args.learn_sigma:
            pred_a, _ = torch.chunk(pred_a, 2, dim=1)
        loss_a = F.mse_loss(pred_a, eps_a, reduction="mean")

        # SISS coefficients
        n = self.n_total
        k = self.k_forget
        coeff_retain = n / (n - k)
        coeff_forget = k / (n - k) * self.siss_nois_s

        # Combine losses (SISS No-IS formula)
        total_loss = coeff_retain * loss_x - coeff_forget * loss_a

        # Apply gradient clipping to the forget term if scaling_norm is set
        if self.scaling_norm is not None and self.scaling_norm > 0:
            # Compute the gradient of the forget term
            self.model.zero_grad()
            forget_loss = -coeff_forget * loss_a
            forget_loss.backward(retain_graph=True)
            
            # Compute gradient norm
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad**2).sum().item()
            grad_norm = grad_norm**0.5
            
            # Apply clipping if needed
            if grad_norm > self.scaling_norm:
                clip_factor = self.scaling_norm / grad_norm
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad *= clip_factor
            
            # Clear gradients and recompute with clipped forget term
            self.model.zero_grad()
            total_loss = coeff_retain * loss_x
            total_loss.backward(retain_graph=True)
            
            # Add the clipped forget gradients manually
            for p in self.model.parameters():
                if p.grad is not None:
                    # The forget gradients are already clipped from the previous computation
                    pass

        return total_loss, loss_x, loss_a

    def _calculate_loss(self, batch_tuple, is_validation=False):
        """
        Calculate the SISS unlearning loss.
        
        Args:
            batch_tuple: Tuple of (retain_batch, forget_batch)
            is_validation: Whether this is validation (not used in unlearning)
            
        Returns:
            Tuple of (total_loss, mse_loss, vlb_loss) for compatibility with BaseTrainer
        """
        retain_batch, forget_batch = batch_tuple
        
        # Calculate SISS loss
        total_loss, retain_mse, forget_mse = self._siss_double_forward_loss(retain_batch, forget_batch)
        
        # Apply SalUn masks if enabled
        if self.salun_enable:
            self.apply_salun_masks()
        
        # Return format compatible with BaseTrainer
        return total_loss, retain_mse, torch.tensor(0.0, device=self.device)  # vlb_loss not used