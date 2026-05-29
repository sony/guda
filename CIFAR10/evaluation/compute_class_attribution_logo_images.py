#!/usr/bin/env python3
"""
Compute per-class CLIP-based attribution scores by comparing images generated
by an unconditional model against images from leave-one-group-out (LOGO) models.

This script assumes that for each image in the unconditional directory, a
corresponding image generated with the same noise/seed exists in each of the
leave-one-group-out directories.

Attribution is defined as `1.0 - CLIP_Score(uncond_img, logo_img)`. A higher
attribution score for a class implies that excluding that class during
training resulted in a significantly different generated image, meaning the
class was important for the original generation.
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import lpips

from evaluation.utils import export_sorted_attribution_scores_from_rows, calculate_ndcg_metrics

def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compute CLIP-based attribution by comparing unconditional and LOGO generated images."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
        help="Dataset to use for determining the number of classes.",
    )
    p.add_argument(
        "--uncond_dir", required=True, help="Folder with images from the model trained on all classes."
    )
    p.add_argument(
        "--logo_base_dir", required=True, help="Base directory containing 'exclude_{class_id}' model outputs."
    )
    p.add_argument(
        "--out_dir", default="./results", help="Directory to save output files."
    )
    p.add_argument(
        "--out_csv", default="class_attribution.csv", help="Output CSV filename for raw scores (metric name will be automatically prepended)."
    )
    p.add_argument(
        "--batch", type=int, default=64, help="Batch size for processing images."
    )
    p.add_argument(
        "--device", default="cuda", help="Computation device (e.g., 'cuda', 'cpu')."
    )
    p.add_argument(
        "--metric", type=str, default="clip", choices=["clip", "lpips", "mse", "all"],
        help="Metric to use for computing attribution scores: 'clip' (1-CLIP similarity), 'lpips' (LPIPS distance), 'mse' (MSE distance), or 'all' (compute all metrics)."
    )
    p.add_argument(
        "--reference_csv", default=None,
        help="Optional: Reference CSV file for calculating nDCG and other ranking metrics."
    )
    return p.parse_args()


@torch.no_grad()
def compute_attribution_for_metric(metric, args, device, num_classes):
    """Compute attribution scores for a specific metric."""
    print(f"\n{'='*50}")
    print(f"Computing attribution scores using {metric.upper()} metric")
    print(f"{'='*50}")
    
    # Add metric name to output filenames
    base_name = args.out_csv.replace('.csv', '')
    out_csv_filename = f"{metric}_{base_name}.csv"
    sorted_csv_filename = f"{metric}_{base_name}_sorted.csv"
    
    out_csv_path = os.path.join(args.out_dir, out_csv_filename)
    sorted_csv_path = os.path.join(args.out_dir, sorted_csv_filename)

    # Initialize models and processors based on the selected metric
    model = None
    processor = None
    lpips_model = None
    transform = None
    
    if metric == "clip":
        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    elif metric == "lpips":
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
        ])
    elif metric == "mse":
        print("Using MSE metric...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # Get paths for unconditionally generated images
    uncond_paths = sorted(Path(args.uncond_dir).glob("image_*.png"))
    assert uncond_paths, f"No PNG images found in {args.uncond_dir}"
    print(f"Found {len(uncond_paths)} unconditional images to process.")

    rows: List[Tuple[str, int, float]] = []

    # Process images in batches to manage memory
    for i in tqdm(range(0, len(uncond_paths), args.batch), desc="Processing Image Batches"):
        batch_uncond_paths = uncond_paths[i : i + args.batch]
        uncond_images = [Image.open(p).convert("RGB") for p in batch_uncond_paths]
        
        # Prepare features based on the selected metric
        if metric == "clip":
            # Get embeddings for the unconditional images in the current batch
            uncond_inputs = processor(text=None, images=uncond_images, return_tensors="pt", padding=True).to(device)
            uncond_feats = model.get_image_features(**uncond_inputs)
            uncond_feats = F.normalize(uncond_feats, dim=1)
        elif metric in ["lpips", "mse"]:
            # Convert images to tensors
            uncond_tensors = torch.stack([transform(img) for img in uncond_images]).to(device)

        # Compare with images from each leave-one-out model
        for class_idx in tqdm(range(num_classes), desc=f"Comparing LOGO classes for batch {i//args.batch}", leave=False):
            logo_img_dir = Path(args.logo_base_dir) / f"exclude_{class_idx}" / "samples"
            if not logo_img_dir.exists():
                print(f"Warning: Directory not found for excluded class {class_idx}, skipping: {logo_img_dir}")
                continue
            
            # Construct paths for the corresponding LOGO images
            batch_logo_paths = [logo_img_dir / p.name for p in batch_uncond_paths]
            logo_images = [Image.open(p).convert("RGB") for p in batch_logo_paths]

            # Calculate attribution scores based on the selected metric
            if metric == "clip":
                # Get embeddings for the LOGO images
                logo_inputs = processor(text=None, images=logo_images, return_tensors="pt", padding=True).to(device)
                logo_feats = model.get_image_features(**logo_inputs)
                logo_feats = F.normalize(logo_feats, dim=1)

                # Calculate cosine similarity between each pair of images (uncond vs LOGO)
                clip_scores = (uncond_feats * logo_feats).sum(dim=1)
                attribution_scores = 1.0 - clip_scores  # Higher attribution = more different
                
            elif metric == "lpips":
                # Convert LOGO images to tensors
                logo_tensors = torch.stack([transform(img) for img in logo_images]).to(device)
                
                # Calculate LPIPS distance (higher = more different)
                lpips_scores = lpips_model(uncond_tensors, logo_tensors).squeeze()
                attribution_scores = lpips_scores  # LPIPS directly gives perceptual distance
                
            elif metric == "mse":
                # Convert LOGO images to tensors
                logo_tensors = torch.stack([transform(img) for img in logo_images]).to(device)
                
                # Calculate MSE distance (higher = more different)
                mse_scores = F.mse_loss(uncond_tensors, logo_tensors, reduction='none')
                mse_scores = mse_scores.view(mse_scores.size(0), -1).mean(dim=1)  # Average over spatial dimensions
                attribution_scores = mse_scores  # MSE directly gives pixel distance

            # Store results
            for path_idx, uncond_path in enumerate(batch_uncond_paths):
                score = attribution_scores[path_idx].item()
                rows.append((str(uncond_path), class_idx, score))

    # Write raw attribution scores to CSV
    print(f"\nSaving raw attribution scores to {out_csv_path}...")
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", f"{metric}_attribution_score"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows.")

    # Export sorted scores
    print(f"Saving sorted scores to {sorted_csv_path}...")
    # Reusing the same utility function as it expects the same (path, class, score) format
    export_sorted_attribution_scores_from_rows(rows, sorted_csv_path, num_classes=num_classes)

    # Calculate ranking metrics if a reference is provided
    if args.reference_csv:
        print(f"\nCalculating ranking metrics against reference CSV for {metric}...")
        m3 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=3, num_classes=num_classes)
        m5 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=5, num_classes=num_classes)
        if isinstance(m3, dict):
            print(f"[nDCG@3] ndcg={m3.get('ndcg@k'):.4f}, top1={m3.get('top1'):.4f}, jaccard={m3.get('jaccard@k'):.4f}")
        if isinstance(m5, dict):
            print(f"[nDCG@5] ndcg={m5.get('ndcg@k'):.4f}, top1={m5.get('top1'):.4f}, jaccard={m5.get('jaccard@k'):.4f}")

    # Clean up models to free memory
    if model is not None:
        del model
        del processor
    if lpips_model is not None:
        del lpips_model
    if transform is not None:
        del transform
    torch.cuda.empty_cache()
    
    print(f"Completed {metric.upper()} metric computation.")
    return sorted_csv_path


@torch.no_grad()
def main() -> None:
    """Main function to execute the script."""
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_classes = 100 if args.dataset == "cifar100" else 10
    print(f"Using dataset: {args.dataset} ({num_classes} classes)")
    print(f"Using metric: {args.metric}")

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.metric == "all":
        # Compute for all metrics
        metrics = ["clip", "lpips", "mse"]
        sorted_csv_paths = []
        
        for metric in metrics:
            sorted_csv_path = compute_attribution_for_metric(metric, args, device, num_classes)
            sorted_csv_paths.append(sorted_csv_path)
        
        print(f"\n{'='*60}")
        print("All metrics completed! Generated files:")
        print(f"{'='*60}")
        for path in sorted_csv_paths:
            print(f"  - {path}")
        
    else:
        # Compute for single metric
        compute_attribution_for_metric(args.metric, args, device, num_classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
