#!/usr/bin/env python3
"""
Pre-compute base model ELBO for all query images.

This cache is used by streaming evaluation during training to avoid
recomputing base model KL divergence hundreds of times.

Workflow:
1. Load base model (trained on all classes)
2. For each batch of images:
   - Set deterministic seed based on image paths
   - Generate noise for all timesteps
   - Compute KL divergence between q(x_{t-1}|x_t,x_0) and p(x_{t-1}|x_t)
3. Save all KL values to .npz file

Storage: ~200-500MB for 2048 images × ~400 timesteps
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from diffusers import UNet2DModel, DDPMScheduler

from cifar10_diffusion.diffusion_utils import DiffusionHelpers, normal_kl
from evaluation.seed_utils import set_seed_for_batch


def get_args():
    p = argparse.ArgumentParser(description="Pre-compute base model ELBO cache")
    p.add_argument("--gen_dir", required=True, help="Directory with generated images")
    p.add_argument("--ckpt_base", required=True, help="Base model checkpoint path")
    p.add_argument("--cache_file", required=True, help="Output .npz cache file path")
    p.add_argument("--batch", type=int, default=128, help="Batch size for processing")
    p.add_argument("--timesteps", type=int, default=4000, help="Total diffusion timesteps")
    p.add_argument("--skip", type=int, default=10, help="Timestep skip interval")
    p.add_argument("--min_t", type=int, default=1, help="Minimum timestep to process")
    p.add_argument("--max_t", type=int, default=None, help="Maximum timestep (None for full range)")
    p.add_argument("--seed", type=int, default=42, help="Base seed for deterministic noise")
    p.add_argument("--device", default="cuda", help="Device to use")
    return p.parse_args()


def load_images(paths, device):
    """Load images and normalize to [-1, 1]"""
    to_tensor = ToTensor()
    imgs = [(to_tensor(Image.open(p).convert("RGB")) * 2 - 1) for p in paths]
    return torch.stack(imgs, dim=0).to(device)


def extract_eps(raw_out, model):
    """Extract noise prediction, handling models with/without learned sigma"""
    num_chan = model.config.in_channels
    if raw_out.shape[1] > num_chan:
        return raw_out[:, :num_chan]
    return raw_out


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("Base ELBO Pre-computation")
    print("="*80)
    print(f"Base model: {args.ckpt_base}")
    print(f"Image directory: {args.gen_dir}")
    print(f"Output cache: {args.cache_file}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    # Load base model
    print("\nLoading base model...")
    base_model = UNet2DModel.from_pretrained(args.ckpt_base).to(device).eval()
    sched = DDPMScheduler.from_pretrained(args.ckpt_base)
    sched.set_timesteps(args.timesteps)
    diffusion_helpers = DiffusionHelpers(sched)
    print(f"✓ Loaded model with {sum(p.numel() for p in base_model.parameters()):,} parameters")
    
    # Get image paths
    paths = sorted(Path(args.gen_dir).glob("image_*.png"))
    if not paths:
        raise ValueError(f"No images found in {args.gen_dir}")
    print(f"✓ Found {len(paths)} images")
    
    # Determine timesteps to process
    max_t = args.max_t if args.max_t else args.timesteps - 1
    timesteps_list = [t for t in reversed(range(0, args.timesteps, args.skip)) 
                     if args.min_t <= t <= max_t]
    print(f"✓ Processing {len(timesteps_list)} timesteps: {timesteps_list[0]}..{timesteps_list[-1]} (skip={args.skip})")
    
    # Initialize storage for KL divergences
    # Shape: (N_images, N_timesteps)
    all_kl = np.zeros((len(paths), len(timesteps_list)), dtype=np.float32)
    
    # Process batches
    print("\nComputing base model KL divergence...")
    for s in tqdm(range(0, len(paths), args.batch), desc="Processing batches"):
        batch_paths = paths[s:s+args.batch]
        
        # Set deterministic seed for this batch
        set_seed_for_batch(batch_paths, args.seed)
        
        # Load images
        x0 = load_images(batch_paths, device)
        B = x0.size(0)
        
        # Pre-generate noise for all timesteps
        eps_for_t = {t: torch.randn_like(x0) for t in timesteps_list if t > 0}
        
        # Compute KL for each timestep
        for t_idx, t_val in enumerate(timesteps_list):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = sched.add_noise(x0, eps_for_t[t_val], t_batch)
            
            # Compute q(x_{t-1} | x_t, x_0) - true posterior
            mu_q, _, log_var_q = diffusion_helpers.q_posterior_mean_variance(
                x_start=x0, x_t=x_t, t=t_batch
            )
            
            # Compute p(x_{t-1} | x_t) - model prediction
            raw_base = base_model(x_t, t_batch).sample
            eps_base = extract_eps(raw_base, base_model)
            x0_hat = diffusion_helpers._predict_xstart_from_eps(x_t, t_batch, eps_base)
            mu_p, _, _ = diffusion_helpers.q_posterior_mean_variance(
                x_start=x0_hat, x_t=x_t, t=t_batch
            )
            
            # Compute KL divergence: D_KL(q || p)
            kl = normal_kl(mu_q.double(), log_var_q.double(), mu_p.double(), log_var_q.double())
            
            # Store per-image KL (already summed over spatial dims by normal_kl)
            # Shape: (B,) already from normal_kl
            all_kl[s:s+B, t_idx] = kl.cpu().numpy()
    
    # Save cache
    print("\nSaving cache...")
    output_dir = Path(args.cache_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        args.cache_file,
        kl_base=all_kl,
        image_paths=np.array([str(p) for p in paths]),
        timesteps=np.array(timesteps_list),
        seed=args.seed,
        config={
            'ckpt_base': args.ckpt_base,
            'gen_dir': args.gen_dir,
            'timesteps': args.timesteps,
            'skip': args.skip,
            'min_t': args.min_t,
            'max_t': max_t
        }
    )
    
    file_size_mb = Path(args.cache_file).stat().st_size / 1024**2
    print(f"✓ Saved cache to {args.cache_file}")
    print(f"  Shape: {all_kl.shape} (N_images × N_timesteps)")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Mean KL: {all_kl.mean():.4f} ± {all_kl.std():.4f}")
    print("\n" + "="*80)
    print("✓ Pre-computation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
