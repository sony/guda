#!/usr/bin/env python3
"""
ELBO computation utilities for Stable Diffusion with LoRA.
Adapted from nga-cifar10 but modified for SD latent space.

Precision Policy:
- UNet forward: BF16 via autocast (for speed)
- KL computation: FP32 (for numerical stability)
- VAE/TextEncoder: FP32 (small compute, stability priority)
- Pre-noise: BF16 (memory reduction)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from peft import PeftModel
from .timing_utils import time_block, get_timing_stats

# Enable TF32 for matrix operations on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, 
              mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussian distributions in FP32.
    
    KL(N(mean1, var1) || N(mean2, var2)) summed across all dimensions except batch.
    
    All inputs must be FP32 for numerical stability.
    
    Args:
        mean1: (B, ...) mean of first distribution (FP32)
        logvar1: (B, ...) log variance of first distribution (FP32)
        mean2: (B, ...) mean of second distribution (FP32)
        logvar2: (B, ...) log variance of second distribution (FP32)
        
    Returns:
        (B,) KL divergence for each batch element (FP32)
    """
    # Ensure FP32
    assert mean1.dtype == torch.float32, f"Expected FP32, got {mean1.dtype}"
    
    # Compute variance from logvar
    var2 = torch.exp(logvar2)
    
    # KL = 0.5 * ((mean1 - mean2)^2 / var2 + exp(logvar1 - logvar2) - 1 + (logvar2 - logvar1))
    kl = 0.5 * (
        ((mean1 - mean2) ** 2) / var2
        + torch.exp(logvar1 - logvar2)
        - 1.0
        + (logvar2 - logvar1)
    )
    # Sum across all dimensions except batch
    return kl.flatten(1).sum(dim=1)


def q_posterior_mean_variance(
    scheduler: DDPMScheduler,
    x_start: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute q(x_{t-1} | x_t, x_0) mean and variance in FP32.
    
    Args:
        scheduler: DDPM scheduler
        x_start: (B, C, H, W) clean latent (FP32)
        x_t: (B, C, H, W) noisy latent at timestep t (FP32)
        t: (B,) timestep indices
        
    Returns:
        mean: (B, C, H, W) posterior mean (FP32)
        variance: (B, 1, 1, 1) posterior variance (FP32)
        log_variance: (B, 1, 1, 1) log posterior variance (FP32)
    """
    # Get alpha values
    alphas_cumprod = scheduler.alphas_cumprod.to(x_start.device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_t_prev = alphas_cumprod[t - 1].view(-1, 1, 1, 1)
    alpha_t_prev = torch.where(t.view(-1, 1, 1, 1) > 0, alpha_t_prev, torch.ones_like(alpha_t_prev))
    
    # Posterior variance
    beta_t = 1 - alpha_t / alpha_t_prev
    variance = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
    variance = torch.clamp(variance, min=1e-20)
    log_variance = torch.log(variance)
    
    # Posterior mean
    # mu = (sqrt(alpha_t_prev) * beta_t * x_0 + sqrt(alpha_t) * (1 - alpha_t_prev) * x_t) / (1 - alpha_t)
    coef_x0 = torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_t_prev) / (1 - alpha_t)
    mean = coef_x0 * x_start + coef_xt * x_t
    
    return mean, variance, log_variance


def predict_x0_from_eps(
    scheduler: DDPMScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor
) -> torch.Tensor:
    """
    Predict x_0 from x_t and predicted noise.
    
    x_0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
    
    Args:
        scheduler: DDPM scheduler
        x_t: (B, C, H, W) noisy latent
        t: (B,) timestep indices
        eps: (B, C, H, W) predicted noise
        
    Returns:
        x_0: (B, C, H, W) predicted clean latent
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    
    x_0 = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
    return x_0


@torch.no_grad()
def compute_elbo_for_latent(
    latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    enable_timing: bool = False
) -> torch.Tensor:
    """
    Compute ELBO for a latent using the variational bound.
    
    ELBO = -sum_t KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))
    
    Precision policy:
    - UNet forward: BF16 via autocast (if use_amp=True)
    - KL computation: FP32 for numerical stability
    - Input/output: FP32
    
    Args:
        latent: (B, C, H, W) clean latent from VAE (FP32)
        prompt_embeds: (B, 77, 768) text embeddings (FP32)
        unet: UNet model (possibly with LoRA)
        scheduler: DDPM scheduler
        timesteps: (T,) timestep values to evaluate
        noise: Optional pre-generated noise (T, B, C, H, W), can be BF16
        num_samples: Number of noise samples to average over
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for AMP (bfloat16 or float16)
        enable_timing: Whether to collect detailed timing statistics
        
    Returns:
        elbo: (B,) ELBO value for each latent (FP32)
    """
    device = latent.device
    B = latent.size(0)
    
    with time_block("elbo.preprocessing", enabled=enable_timing):
        # Ensure input is FP32
        latent = latent.float()
        prompt_embeds = prompt_embeds.float()
        
        # Apply channels_last for better memory bandwidth
        latent = latent.to(memory_format=torch.channels_last)
        
        # Ensure prompt_embeds matches batch size
        if prompt_embeds.size(0) != B:
            if prompt_embeds.size(0) == 1:
                prompt_embeds = prompt_embeds.expand(B, -1, -1).contiguous()
            else:
                raise ValueError(f"prompt_embeds batch size {prompt_embeds.size(0)} doesn't match latent batch size {B}")
    
    elbo_sum = torch.zeros(B, device=device, dtype=torch.float32)
    
    for sample_idx in range(num_samples):
        for t_idx, t_val in enumerate(timesteps):
            with time_block("elbo.timestep_setup", enabled=enable_timing):
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                
                # Use pre-generated noise if available, otherwise generate new
                if noise is not None:
                    # noise shape: (num_samples, T, B, C, H, W) or (T, B, C, H, W)
                    if noise.ndim == 6:  # (num_samples, T, B, C, H, W)
                        eps_t = noise[sample_idx, t_idx].float()
                    else:  # (T, B, C, H, W)
                        eps_t = noise[t_idx].float()
                else:
                    eps_t = torch.randn_like(latent, dtype=torch.float32)
            
            with time_block("elbo.forward_process", enabled=enable_timing):
                # Forward process: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps (FP32)
                x_t = scheduler.add_noise(latent, eps_t, t_batch)
            
            with time_block("elbo.q_posterior", enabled=enable_timing):
                # q(x_{t-1} | x_t, x_0) - true posterior (FP32)
                mu_q, _, log_var_q = q_posterior_mean_variance(scheduler, latent, x_t, t_batch)
            
            # p(x_{t-1} | x_t) - model prediction
            # UNet forward in BF16 for speed
            with time_block("elbo.unet_forward", enabled=enable_timing):
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        noise_pred = unet(x_t, t_batch, encoder_hidden_states=prompt_embeds).sample
                    noise_pred = noise_pred.float()  # Convert back to FP32
                else:
                    noise_pred = unet(x_t, t_batch, encoder_hidden_states=prompt_embeds).sample
            
            with time_block("elbo.p_posterior", enabled=enable_timing):
                # Predict x0 and compute model posterior (FP32)
                x0_pred = predict_x0_from_eps(scheduler, x_t, t_batch, noise_pred)
                mu_p, _, _ = q_posterior_mean_variance(scheduler, x0_pred, x_t, t_batch)
            
            with time_block("elbo.kl_divergence", enabled=enable_timing):
                # KL divergence (FP32, use same variance for both distributions)
                kl = normal_kl(mu_q, log_var_q, mu_p, log_var_q)
                
                # ELBO = -KL (negative because we want to maximize ELBO)
                elbo_sum += -kl
    
    # Average over samples
    return elbo_sum / num_samples


@torch.no_grad()
def compute_delta_elbo(
    latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    unet_base: UNet2DConditionModel,
    unet_modified: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    min_t: int = 0,
    max_t: int = 999,
    skip_t: int = 10,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Compute ΔELBO = ELBO(modified) - ELBO(base).
    
    Uses the same noise for both models to reduce variance.
    
    Args:
        latent: (B, C, H, W) clean latent
        prompt_embeds: (B, 77, 768) text embeddings
        unet_base: Base UNet (All-class model)
        unet_modified: Modified UNet (LOGO or Unlearned model)
        scheduler: DDPM scheduler  
        min_t: Minimum timestep
        max_t: Maximum timestep
        skip_t: Timestep interval
        num_samples: Number of noise samples to average
        
    Returns:
        delta_elbo: (B,) ΔELBO for each latent
    """
    device = latent.device
    B = latent.size(0)
    
    # Timesteps to evaluate
    timesteps = torch.tensor(
        list(range(min_t, max_t + 1, skip_t)),
        device=device,
        dtype=torch.long
    )
    
    delta_sum = torch.zeros(B, device=device)
    
    for sample_idx in range(num_samples):
        # Pre-generate noise for this sample (same for both models)
        noise = torch.stack([torch.randn_like(latent) for _ in timesteps])
        
        # Compute ELBO for both models with same noise
        elbo_base = compute_elbo_for_latent(
            latent, prompt_embeds, unet_base, scheduler, timesteps, noise, num_samples=1
        )
        elbo_modified = compute_elbo_for_latent(
            latent, prompt_embeds, unet_modified, scheduler, timesteps, noise, num_samples=1
        )
        
        delta_sum += (elbo_modified - elbo_base)
    
    return delta_sum / num_samples


@torch.no_grad()
def precompute_elbo_for_latent(
    latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    num_samples: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    enable_timing: bool = False
) -> torch.Tensor:
    """
    Precompute ELBO for a single model (typically all-class model).
    This can be cached and reused when computing delta ELBO with multiple models.
    
    Args:
        latent: (B, C, H, W) clean latent (FP32)
        prompt_embeds: (B, 77, 768) text embeddings (FP32)
        unet: UNet model
        scheduler: DDPM scheduler
        timesteps: (T,) timestep values to evaluate
        noise: (T, B, C, H, W) pre-generated noise (can be BF16)
        num_samples: Number of noise samples (should be 1 when caching)
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for AMP
        enable_timing: Whether to collect detailed timing statistics
        
    Returns:
        elbo: (B,) ELBO value for each latent (FP32)
    """
    return compute_elbo_for_latent(
        latent, prompt_embeds, unet, scheduler, timesteps, noise, 
        num_samples=num_samples, use_amp=use_amp, amp_dtype=amp_dtype,
        enable_timing=enable_timing
    )


@torch.no_grad()
def compute_delta_elbo_with_cache(
    latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    unet_base: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    pre_noise: torch.Tensor,
    elbo_modified_cache: torch.Tensor,
    num_samples: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    enable_timing: bool = False
) -> torch.Tensor:
    """
    Compute ΔELBO = ELBO(modified) - ELBO(base) using cached ELBO for modified model.
    
    This is optimized for the case where the modified model (e.g., all-class) is the same
    across multiple base models (e.g., different LOGO models). The modified model's ELBO
    is precomputed once and cached.
    
    Args:
        latent: (B, C, H, W) clean latent (FP32)
        prompt_embeds: (B, 77, 768) text embeddings (FP32)
        unet_base: Base UNet (typically LOGO or Unlearned model)
        scheduler: DDPM scheduler
        timesteps: (T,) timestep values to evaluate
        pre_noise: (num_samples, T, B, C, H, W) pre-generated noise (can be BF16)
        elbo_modified_cache: (B,) pre-computed ELBO for modified model (FP32)
        num_samples: Number of noise samples to average
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for AMP
        enable_timing: Whether to collect detailed timing statistics
        
    Returns:
        delta_elbo: (B,) ΔELBO for each latent (FP32)
    """
    device = latent.device
    B = latent.size(0)
    
    delta_sum = torch.zeros(B, device=device, dtype=torch.float32)
    
    for sample_idx in range(num_samples):
        # Extract noise for this sample: (T, B, C, H, W)
        noise = pre_noise[sample_idx]
        
        # Compute ELBO only for base model (modified is cached)
        elbo_base = compute_elbo_for_latent(
            latent, prompt_embeds, unet_base, scheduler, timesteps, noise, 
            num_samples=1, use_amp=use_amp, amp_dtype=amp_dtype,
            enable_timing=enable_timing
        )
        
        # Use cached ELBO for modified model (already FP32)
        delta_sum += (elbo_modified_cache - elbo_base)
    
    return delta_sum / num_samples


@torch.no_grad()
def compute_delta_elbo_batch(
    latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    unet_base: UNet2DConditionModel,
    unet_modified: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    pre_noise: torch.Tensor,
    num_samples: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Compute ΔELBO = ELBO(modified) - ELBO(base) using pre-generated noise.
    
    This is an optimized version that uses pre-generated noise for consistency
    across different models when processing the same images multiple times.
    
    Args:
        latent: (B, C, H, W) clean latent (FP32)
        prompt_embeds: (B, 77, 768) text embeddings (FP32)
        unet_base: Base UNet (typically LOGO model)
        unet_modified: Modified UNet (typically All-class model)
        scheduler: DDPM scheduler
        timesteps: (T,) timestep values to evaluate
        pre_noise: (num_samples, T, B, C, H, W) pre-generated noise (can be BF16)
        num_samples: Number of noise samples to average
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for AMP
        
    Returns:
        delta_elbo: (B,) ΔELBO for each latent (FP32)
    """
    device = latent.device
    B = latent.size(0)
    
    delta_sum = torch.zeros(B, device=device, dtype=torch.float32)
    
    for sample_idx in range(num_samples):
        # Extract noise for this sample: (T, B, C, H, W)
        noise = pre_noise[sample_idx]
        
        # Compute ELBO for both models with same noise
        elbo_base = compute_elbo_for_latent(
            latent, prompt_embeds, unet_base, scheduler, timesteps, noise, 
            num_samples=1, use_amp=use_amp, amp_dtype=amp_dtype
        )
        elbo_modified = compute_elbo_for_latent(
            latent, prompt_embeds, unet_modified, scheduler, timesteps, noise, 
            num_samples=1, use_amp=use_amp, amp_dtype=amp_dtype
        )
        
        delta_sum += (elbo_modified - elbo_base)
    
    return delta_sum / num_samples
