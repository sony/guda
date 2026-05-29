#!/usr/bin/env python3
"""
Compute per-class CLIP-based attribution scores for generated images.
Works with both CIFAR-10 and CIFAR-100.

Usage
-----
# For CIFAR-10
python evaluation/compute_clip_class_attribution.py \
    --gen_dir ./tmp_gen \
    --out_dir ./results \
    --out_csv clip_baseline_cifar10.csv \
    --dataset cifar10 \
    --num_samples 1000 \
    --batch 64

# For CIFAR-100
python evaluation/compute_clip_class_attribution.py \
    --gen_dir /path/to/cifar100_images \
    --out_dir ./results_cifar100 \
    --out_csv clip_baseline_cifar100.csv \
    --dataset cifar100 \
    --num_samples 100 \
    --batch 32
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision.datasets import CIFAR10, CIFAR100 
from tqdm.auto import tqdm

from evaluation.data_manifests import (
    TrainManifest,
    load_train_manifest_checked,
    select_first_train_ids_per_group,
)
from evaluation.utils import export_sorted_attribution_scores_from_rows, calculate_ndcg_metrics

def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compute CLIP-based attribution for generated CIFAR images"
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to use for attribution (cifar10 or cifar100)."
    )
    p.add_argument(
        "--metric",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity metric: 'cosine' (normalized) or 'dot' (unnormalized inner product)"
    )
    p.add_argument(
        "--gen_dir", required=True, help="Folder with generated images (e.g., image_*.png)"
    )
    p.add_argument(
        "--out_dir", default="./results", help="Directory to save output files"
    )
    p.add_argument(
        "--out_csv", default="clip_baseline.csv", help="Output CSV filename for raw scores"
    )
    p.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of dataset train samples per class to use for embeddings."
    )
    p.add_argument(
        "--batch", type=int, default=64, help="Batch size for processing generated images"
    )
    p.add_argument(
        "--device", default="cuda", help="Computation device (e.g., 'cuda', 'cpu')"
    )
    p.add_argument(
        "--reference_csv", default=None,
        help="Optional: Reference CSV file for calculating nDCG and other ranking metrics."
    )
    p.add_argument(
        "--train_manifest", type=str, default=None,
        help="Optional train manifest whose group_ids define the reference groups."
    )
    return p.parse_args()


def collect_reference_samples_by_group(
    *,
    dataset,
    num_groups: int,
    num_samples_per_group: int,
    train_manifest: Optional[TrainManifest] = None,
) -> Dict[int, List[Image.Image]]:
    """
    Collect reference train samples per group.

    Without a manifest, groups follow the dataset's raw labels and samples are taken
    in dataset order (backward-compatible behavior). With a manifest, groups follow
    manifest.group_ids and train ids are selected in manifest order.
    """
    group_samples: Dict[int, List[Image.Image]] = {i: [] for i in range(num_groups)}

    if train_manifest is None:
        targets = dataset.targets
        for train_id, label in enumerate(tqdm(targets, desc="Collecting samples")):
            label = int(label)
            if len(group_samples[label]) < num_samples_per_group:
                group_samples[label].append(dataset[train_id][0])
            if all(len(samples) >= num_samples_per_group for samples in group_samples.values()):
                break
        return group_samples

    selected_ids = select_first_train_ids_per_group(
        train_ids=train_manifest.train_ids,
        group_ids=train_manifest.group_ids,
        num_groups=num_groups,
        num_samples_per_group=num_samples_per_group,
    )
    for group_id in range(num_groups):
        group_samples[group_id] = [dataset[train_id][0] for train_id in selected_ids[group_id]]
    return group_samples

@torch.no_grad()
def main() -> None:
    """Main function to execute the script."""
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine dataset-specific parameters
    if args.dataset == "cifar100":
        DatasetClass = CIFAR100
        num_classes = 100
    else:  # default to cifar10
        DatasetClass = CIFAR10
        num_classes = 10
    
    print(f"Using dataset: {args.dataset} ({num_classes} classes)")

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Prepare output file paths
    out_csv_path = os.path.join(args.out_dir, args.out_csv)
    sorted_csv_name = args.out_csv.replace('.csv', '_sorted.csv')
    sorted_csv_path = os.path.join(args.out_dir, sorted_csv_name)

    # Load CLIP model and processor
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()

    # Prepare dataset training samples per class
    print(f"Loading {args.dataset} training samples ({args.num_samples} per class)...")
    dataset = DatasetClass(root="./data", train=True, download=True)
    train_manifest = None
    if args.train_manifest:
        train_manifest = load_train_manifest_checked(
            Path(args.train_manifest),
            dataset_name=args.dataset,
            split="train",
            expected_len=len(dataset),
        )
        print(f"Using train manifest groups from {args.train_manifest}")

    class_samples = collect_reference_samples_by_group(
        dataset=dataset,
        num_groups=num_classes,
        num_samples_per_group=args.num_samples,
        train_manifest=train_manifest,
    )

    # Check if all classes have enough samples
    for c, samples in class_samples.items():
        if len(samples) < args.num_samples:
            print(f"Warning: Class {c} has only {len(samples)} samples, less than the requested {args.num_samples}.")

    # Precompute CLIP embeddings for each class from the dataset
    print(f"Computing CLIP embeddings for {args.dataset} samples...")
    dataset_embeddings: Dict[int, torch.Tensor] = {}
    for cls in tqdm(range(num_classes), desc="Computing class embeddings"):
        imgs = class_samples[cls]
        if not imgs:
            print(f"Warning: No images found for class {cls}, skipping.")
            continue
            
        # Process images in batches to avoid memory issues with large num_samples
        class_feats_list = []
        embedding_batch_size = 512  # Batch size for embedding calculation
        for i in range(0, len(imgs), embedding_batch_size):
            batch_imgs = imgs[i : i + embedding_batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feats = outputs.pooler_output
            class_feats_list.append(feats.cpu())
        
        class_feats = torch.cat(class_feats_list, dim=0).to(device)

        if args.metric == "cosine":
            class_feats = F.normalize(class_feats, dim=1)
        dataset_embeddings[cls] = class_feats

    # Process generated images
    paths = sorted(Path(args.gen_dir).glob("image_*.png"))
    assert paths, f"No PNG images found in {args.gen_dir}"
    rows: List[Tuple[str, int, float]] = []

    print(f"Computing CLIP-based scores for {len(paths)} generated images...")
    for i in tqdm(range(0, len(paths), args.batch), desc="Processing generated images"):
        batch_paths = paths[i : i + args.batch]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        gen_feats = outputs.pooler_output  # [B, embed_dim]
        if args.metric == "cosine":
            gen_feats = F.normalize(gen_feats, dim=1)

        # Compute average similarity per class for each image in the batch
        for idx, p in enumerate(batch_paths):
            f = gen_feats[idx]  # [embed_dim]
            for cls in range(num_classes):
                if cls not in dataset_embeddings:
                    continue  # Skip classes with no reference embeddings
                # Inner product is equivalent to cosine similarity for normalized vectors
                sims = (dataset_embeddings[cls] @ f).squeeze()  # [num_samples]
                score = sims.mean().item()
                rows.append((str(p), cls, score))

    # Write raw scores to CSV
    print(f"\nSaving raw scores to {out_csv_path}...")
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "clip_score"])  # Changed 'delta_elbo' to 'clip_score' for clarity
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows.")

    # Export sorted scores and compute nDCG if a reference is provided
    # Note: `export_sorted_attribution_scores_from_rows` in the provided `utils.py` uses hardcoded CIFAR-10 class names.
    # The output `_sorted.csv` will have incorrect class names in the header for CIFAR-100, but the numeric values will be correct.
    print(f"Saving sorted scores to {sorted_csv_path}...")
    export_sorted_attribution_scores_from_rows(rows, sorted_csv_path, num_classes=num_classes)

    if args.reference_csv:
        print("\nCalculating ranking metrics against reference CSV...")
        m3 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=3, num_classes=num_classes)
        m5 = calculate_ndcg_metrics(sorted_csv_path, args.reference_csv, k=5, num_classes=num_classes)
        if isinstance(m3, dict):
            print(f"[nDCG@3] ndcg={m3.get('ndcg@k'):.4f}, top1={m3.get('top1'):.4f}")
        if isinstance(m5, dict):
            print(f"[nDCG@5] ndcg={m5.get('ndcg@k'):.4f}, top1={m5.get('top1'):.4f}")

    print("\n Done.")

if __name__ == "__main__":
    main()
