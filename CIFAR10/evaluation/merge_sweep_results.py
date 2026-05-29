#!/usr/bin/env python3
"""
Merge delta ELBO results from individual class unlearning experiments.

For each hyperparameter configuration (lambda × lr), merges CSVs from 
class_0 through class_9 directories to create a complete attribution table.

Input structure:
  outputs/hyperparam_sweep/{method}_cifar10/{config}/class_{0-9}/delta_elbo_epoch_XXXX.csv

Output structure:
  outputs/hyperparam_sweep/{method}_cifar10/{config}/merged/delta_elbo_epoch_XXXX_merged.csv
  outputs/hyperparam_sweep/{method}_cifar10/{config}/merged/delta_elbo_epoch_XXXX_merged_sorted.csv
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def read_class_csv(csv_path: Path, target_class: int) -> Dict[str, float]:
    """
    Read a single class CSV and extract scores for the target class.
    
    CSV format: image, class_id, delta_elbo
    
    Args:
        csv_path: Path to CSV file
        target_class: Class ID to extract scores for
    
    Returns:
        Dict mapping image_name -> score for the target class
    """
    scores = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['class_id'])
            if class_id == target_class:
                # Extract just the filename without full path
                image_path = row['image']
                image = image_path.split('/')[-1]  # Get just filename
                score = float(row['delta_elbo'])
                scores[image] = score
    return scores


def merge_config_epoch(config_dir: Path, epoch: str, num_classes: int = 10) -> None:
    """
    Merge results for a single epoch across all classes.
    
    Args:
        config_dir: Path to config directory (e.g., lambda0.003_lr1e-5/)
        epoch: Epoch string (e.g., "0010", "0020")
        num_classes: Number of classes (10 for CIFAR-10)
    """
    # Read all class CSVs
    class_scores = {}  # class_id -> {image -> score}
    image_list = None
    
    for class_id in range(num_classes):
        class_dir = config_dir / f"class_{class_id}"
        csv_path = class_dir / f"delta_elbo_epoch_{epoch}.csv"
        
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping class {class_id}")
            continue
        
        scores = read_class_csv(csv_path, class_id)
        class_scores[class_id] = scores
        
        # Use first valid CSV to get image list
        if image_list is None and scores:
            image_list = sorted(scores.keys())
    
    if image_list is None:
        print(f"  Error: No valid CSVs found for epoch {epoch}")
        return
    
    # Create merged output directory
    merged_dir = config_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    
    # Build merged data
    merged_data = []
    for image in image_list:
        row = {'image': image}
        for class_id in range(num_classes):
            score = class_scores.get(class_id, {}).get(image, 0.0)
            row[f'R{class_id+1}_Cls'] = class_id
            row[f'R{class_id+1}_Name'] = f"{class_id}: {CLASS_NAMES[class_id]}"
            row[f'R{class_id+1}_Sc'] = score
        merged_data.append(row)
    
    # Write merged CSV
    output_path = merged_dir / f"delta_elbo_epoch_{epoch}_merged.csv"
    fieldnames = ['image']
    for i in range(num_classes):
        fieldnames.extend([f'R{i+1}_Cls', f'R{i+1}_Name', f'R{i+1}_Sc'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)
    
    print(f"  ✓ Merged: {output_path.name}")
    
    # Create sorted version
    sorted_data = []
    for row in merged_data:
        image = row['image']
        # Extract scores and sort
        scores = [(row[f'R{i+1}_Cls'], row[f'R{i+1}_Name'], row[f'R{i+1}_Sc']) 
                  for i in range(num_classes)]
        scores.sort(key=lambda x: x[2], reverse=True)  # Sort by score descending
        
        sorted_row = {'image': image}
        for rank, (cls, name, score) in enumerate(scores, 1):
            sorted_row[f'R{rank}_Cls'] = cls
            sorted_row[f'R{rank}_Name'] = name
            sorted_row[f'R{rank}_Sc'] = score
        sorted_data.append(sorted_row)
    
    sorted_path = merged_dir / f"delta_elbo_epoch_{epoch}_merged_sorted.csv"
    with open(sorted_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)
    
    print(f"  ✓ Sorted: {sorted_path.name}")


def merge_all_configs(sweep_dir: Path, epochs: List[str]) -> None:
    """
    Merge results for all configurations in a sweep directory.
    
    Args:
        sweep_dir: Path to sweep directory (e.g., outputs/hyperparam_sweep/retrack_cifar10/)
        epochs: List of epoch strings to process (e.g., ["0010", "0020", ...])
    """
    # Find all config directories
    config_dirs = [d for d in sweep_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('lambda')]
    
    print(f"\nFound {len(config_dirs)} configurations in {sweep_dir.name}/")
    
    for config_dir in sorted(config_dirs):
        print(f"\n{config_dir.name}/")
        for epoch in epochs:
            merge_config_epoch(config_dir, epoch)


def main():
    parser = argparse.ArgumentParser(description="Merge sweep results across classes")
    parser.add_argument("--sweep_dir", required=True, 
                        help="Path to sweep directory (e.g., outputs/hyperparam_sweep/retrack_cifar10)")
    parser.add_argument("--epochs", nargs='+', default=['0010', '0020', '0030', '0040', '0050'],
                        help="List of epoch strings to process")
    parser.add_argument("--method", choices=['retrack', 'esd'], 
                        help="Method name (auto-detected from sweep_dir if not specified)")
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return
    
    # Auto-detect method from directory name
    if args.method is None:
        if 'retrack' in sweep_dir.name.lower():
            method = 'retrack'
        elif 'esd' in sweep_dir.name.lower():
            method = 'esd'
        else:
            print(f"Warning: Could not auto-detect method from {sweep_dir.name}")
            method = 'unknown'
    else:
        method = args.method
    
    print("="*80)
    print(f"Merging Sweep Results: {method.upper()}")
    print("="*80)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Epochs to process: {', '.join(args.epochs)}")
    
    merge_all_configs(sweep_dir, args.epochs)
    
    print("\n" + "="*80)
    print("Merging complete!")
    print("="*80)


if __name__ == "__main__":
    main()
