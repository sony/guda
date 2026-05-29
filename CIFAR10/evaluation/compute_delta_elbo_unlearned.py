#!/usr/bin/env python3
"""
Compute per-class ΔELBO using unlearned models for CIFAR-10/100 samples.
This script compares predictions from an unconditional base model with 
unlearned models (ESD or similar) where specific classes have been unlearned.
Memory-efficient by loading one model at a time.
"""
import argparse
import csv
import os
import math
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from cifar10_diffusion.diffusion_utils import DiffusionHelpers, normal_kl
from evaluation.utils import export_sorted_attribution_scores_from_rows, calculate_ndcg_metrics
from evaluation.seed_utils import set_seed_for_batch

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
        help="Dataset to use, determines the number of classes."
    )
    p.add_argument("--gen_dir", required=True, help="Folder with image_*.png")
    p.add_argument("--ckpt_base", required=True, help="Checkpoint for base unconditional model")
    p.add_argument("--ckpt_unlearned", required=True, help="Base directory containing unlearn_class_X models")
    p.add_argument("--ckpt_epoch", type=str, default="latest", help="Epoch directory name of the checkpoint.")
    p.add_argument("--out_csv", default="delta_elbo_unlearned.csv", help="Output CSV filename")
    p.add_argument("--output_dir", default="./", help="Directory to store output CSV files")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--skip", type=int, default=10, help="timesteps to skip")
    p.add_argument("--min_t", type=int, default=1, help="Minimum timestep to process")
    p.add_argument("--max_t", type=int, default=None, help="Maximum timestep to process (None for full range)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference_csv", default=None, help="Reference CSV for NDCG calculation")
    p.add_argument("--seed", type=int, default=42, help="Base seed for deterministic noise generation")
    p.add_argument("--base_elbo_cache", default=None, 
                   help="Path to pre-computed base ELBO .npz file (optional, for faster evaluation)")
    p.add_argument("--single_class_mode", action="store_true",
                   help="Evaluate single class model (ckpt_unlearned is model dir, not parent dir)")
    return p.parse_args()

def load_images(paths: List[Path], device: torch.device) -> torch.Tensor:
    """Load PNGs, scale to [-1,1]. Shape: (B,3,32,32)."""
    to_tensor = ToTensor()
    imgs = [(to_tensor(Image.open(p).convert("RGB")) * 2 - 1) for p in paths]
    return torch.stack(imgs, dim=0).to(device)

def extract_eps(raw_out: torch.Tensor, model: UNet2DModel) -> torch.Tensor:
    """Extract noise prediction, handling models with/without learned sigma."""
    num_chan = model.config.in_channels
    if raw_out.shape[1] > num_chan:
        return raw_out[:, :num_chan]
    return raw_out


@torch.no_grad()
def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_classes = 100 if args.dataset == "cifar100" else 10
    print(f"Running for {args.dataset} with {num_classes} classes.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = output_dir / args.out_csv
    
    # --- Load base unconditional model (before unlearning) once ---
    print("Loading base unconditional model...")
    base_model = UNet2DModel.from_pretrained(args.ckpt_base).to(device).eval()
    sched = DDPMScheduler.from_pretrained(args.ckpt_base)
    sched.set_timesteps(args.timesteps)
    diffusion_helpers = DiffusionHelpers(sched)
    
    # --- Load pre-computed base ELBO cache if available ---
    base_elbo_cache = None
    if args.base_elbo_cache and os.path.exists(args.base_elbo_cache):
        print(f"Loading base ELBO cache from {args.base_elbo_cache}")
        import numpy as np
        cache_data = np.load(args.base_elbo_cache)
        base_elbo_cache = {
            'kl_base': torch.from_numpy(cache_data['kl_base']),
            'image_paths': [str(p) for p in cache_data['image_paths']],
            'timesteps': cache_data['timesteps'].tolist()
        }
        print(f"  Loaded cache for {len(base_elbo_cache['image_paths'])} images, "
              f"{len(base_elbo_cache['timesteps'])} timesteps")
        print(f"  Cache seed: {cache_data['seed']}")
    
    paths: List[Path] = sorted(Path(args.gen_dir).glob("image_*.png"))
    assert paths, f"No PNG images found in {args.gen_dir}"

    # Intermediate storage for delta scores for each image
    delta_scores = defaultdict(lambda: torch.zeros(num_classes, device='cpu'))
    
    # Determine timestep range
    max_timestep = args.max_t if args.max_t is not None else args.timesteps - 1
    timesteps_to_process = [t for t in reversed(range(0, args.timesteps, args.skip)) 
                           if args.min_t <= t <= max_timestep]

    # --- Process each batch of images ---
    for s in tqdm(range(0, len(paths), args.batch), desc="Processing Batches"):
        batch_paths = paths[s:s+args.batch]
        
        # Set deterministic seed for this batch
        set_seed_for_batch(batch_paths, args.seed)
        
        x0 = load_images(batch_paths, device)
        B = x0.size(0)

        # 1. Pre-generate noise for all timesteps for this batch
        eps_for_t = {t: torch.randn_like(x0) for t in timesteps_to_process if t > 0}
        
        # 2. Pre-calculate KL divergence for the 'base_model' for all timesteps
        kl_base_per_t = {}
        
        if base_elbo_cache is not None:
            # Use cached KL values
            for t_idx, t_val in enumerate(timesteps_to_process):
                # Map batch paths to cache indices
                batch_indices = [base_elbo_cache['image_paths'].index(str(p)) for p in batch_paths]
                kl_base_per_t[t_val] = torch.stack([
                    base_elbo_cache['kl_base'][idx, t_idx] for idx in batch_indices
                ]).to(device)
        else:
            # Compute on-the-fly (original behavior)
            for t_val in timesteps_to_process:
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                x_t = sched.add_noise(x0, eps_for_t[t_val], t_batch)
                
                mu_q, _, log_var_q = diffusion_helpers.q_posterior_mean_variance(x_start=x0, x_t=x_t, t=t_batch)
                
                raw_base = base_model(x_t, t_batch).sample
                eps_base = extract_eps(raw_base, base_model)
                x0_hat_base = diffusion_helpers._predict_xstart_from_eps(x_t, t_batch, eps_base)
                mu_p_base, _, _ = diffusion_helpers.q_posterior_mean_variance(x_start=x0_hat_base, x_t=x_t, t=t_batch)
                
                kl_base = normal_kl(mu_q.double(), log_var_q.double(), mu_p_base.double(), log_var_q.double())
                kl_base_per_t[t_val] = kl_base.to(device)

        # 3. Loop through each class, load the corresponding unlearned model, and compute delta
        # 3. Determine which classes to process
        if args.single_class_mode:
            # Single class mode: ckpt_unlearned is the model directory itself
            # Extract class index from path (e.g., .../class_3/epoch_0010 -> class_idx=3)
            import re
            match = re.search(r'class_(\d+)', args.ckpt_unlearned)
            if match:
                class_idx = int(match.group(1))
                classes_to_process = [(class_idx, args.ckpt_unlearned)]
            else:
                # Fallback: process as class 0
                print("Warning: Could not extract class index from path, using class_idx=0")
                classes_to_process = [(0, args.ckpt_unlearned)]
        else:
            # Multi-class mode: ckpt_unlearned is parent directory with unlearn_class_X subdirs
            classes_to_process = []
            for class_idx in range(num_classes):
                unlearned_path = os.path.join(args.ckpt_unlearned, f"unlearn_class_{class_idx}", args.ckpt_epoch)
                if os.path.exists(unlearned_path):
                    classes_to_process.append((class_idx, unlearned_path))
                else:
                    print(f"Warning: Skipping unlearn_class_{class_idx}, path not found: {unlearned_path}")
        
        # 4. Loop through classes and compute delta
        for class_idx, unlearned_path in tqdm(classes_to_process, desc="Processing Classes", leave=False):
            unlearned_model = UNet2DModel.from_pretrained(unlearned_path).to(device).eval()
            
            delta_for_this_class = torch.zeros(B, device=device)

            for t_val in timesteps_to_process:
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                x_t = sched.add_noise(x0, eps_for_t[t_val], t_batch)
                
                mu_q, _, log_var_q = diffusion_helpers.q_posterior_mean_variance(x_start=x0, x_t=x_t, t=t_batch)

                raw_unlearned = unlearned_model(x_t, t_batch).sample
                eps_unlearned = extract_eps(raw_unlearned, unlearned_model)
                x0_hat_unlearned = diffusion_helpers._predict_xstart_from_eps(x_t, t_batch, eps_unlearned)
                mu_p_unlearned, _, _ = diffusion_helpers.q_posterior_mean_variance(x_start=x0_hat_unlearned, x_t=x_t, t=t_batch)
                
                kl_unlearned = normal_kl(mu_q.double(), log_var_q.double(), mu_p_unlearned.double(), log_var_q.double())
                
                # Retrieve pre-calculated kl_base and compute delta for this timestep
                # ΔELBO = KL(q||p_unlearned) - KL(q||p_base)
                # Positive values indicate higher likelihood under base model (class was important)
                delta_for_this_class += (kl_unlearned.to(device) - kl_base_per_t[t_val]).to(torch.float32)
            
            # Store the final delta score for this class and batch
            for i, p in enumerate(batch_paths):
                delta_scores[str(p)][class_idx] = delta_for_this_class[i].item()

            # Unload the model to free up GPU memory
            del unlearned_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Convert defaultdict to final rows for CSV writing ---
    rows = []
    sorted_paths = sorted(delta_scores.keys())
    for p in sorted_paths:
        scores = delta_scores[p]
        for c in range(num_classes):
            rows.append((str(p), c, scores[c].item()))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "delta_elbo"])
        writer.writerows(rows)
    print(f"\nSaved final results to {out_csv_path} ({len(rows)} rows)")

    sorted_csv_path = str(out_csv_path).replace('.csv', '_sorted.csv')
    export_sorted_attribution_scores_from_rows(rows, sorted_csv_path, num_classes=num_classes)
    print(f"Saved sorted results to {sorted_csv_path}")

    if args.reference_csv:
        metrics3 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=3, num_classes=num_classes)
        metrics5 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=5, num_classes=num_classes)
        # Brief log
        if isinstance(metrics3, dict):
            print(f"[nDCG@3] ndcg={metrics3.get('ndcg@k'):.4f}, top1={metrics3.get('top1'):.4f}, spearman_val={metrics3.get('spearman_value'):.4f}")
        if isinstance(metrics5, dict):
            print(f"[nDCG@5] ndcg={metrics5.get('ndcg@k'):.4f}, top1={metrics5.get('top1'):.4f}, spearman_val={metrics5.get('spearman_value'):.4f}")

if __name__ == "__main__":
    main()