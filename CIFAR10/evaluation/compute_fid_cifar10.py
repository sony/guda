#!/usr/bin/env python3
# compute_fid_cifar10.py

import argparse
import glob
import os
from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID on each generated-folder using CIFAR-10 train statistics"
    )
    parser.add_argument(
        "--base_dir", type=str, default="cifar10_generated",
        help="Directory containing generated subfolders or images"
    )
    parser.add_argument(
        "--pattern", type=str, default=None,
        help="Glob pattern to match subdirectories under base_dir. If not specified, base_dir itself is processed."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="PyTorch device for FID computation"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10",
        help="Dataset name for built-in statistics"
    )
    parser.add_argument(
        "--dataset_res", type=int, default=32,
        help="Resolution of images"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train",
        help="Which split to use: 'train' (50k) or 'test' (10k)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # If pattern is specified, get matching folders
    if args.pattern is not None:
        folders = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))
        if not folders:
            print(f"No folders match {args.base_dir}/{args.pattern}")
            return
            
        # Process each matching folder
        for gen_dir in folders:
            name = os.path.basename(gen_dir)
            print(f"\n>>> Computing FID for {name} <<<")
            score = fid.compute_fid(
                gen_dir,
                dataset_name=args.dataset_name,
                dataset_res=args.dataset_res,
                dataset_split=args.dataset_split,
                device=args.device
            )
            print(f"{name} → FID: {score:.4f}")
    else:
        # Process base_dir directly
        name = os.path.basename(args.base_dir)
        print(f"\n>>> Computing FID for {name} (direct) <<<")
        
        # Check if the directory contains images
        image_files = glob.glob(os.path.join(args.base_dir, "*.png")) + \
                     glob.glob(os.path.join(args.base_dir, "*.jpg")) + \
                     glob.glob(os.path.join(args.base_dir, "*.jpeg"))
        
        if not image_files:
            print(f"Warning: No image files found in {args.base_dir}")
            return
            
        print(f"Found {len(image_files)} images in {args.base_dir}")
        
        score = fid.compute_fid(
            args.base_dir,
            dataset_name=args.dataset_name,
            dataset_res=args.dataset_res,
            dataset_split=args.dataset_split,
            device=args.device
        )
        print(f"{name} → FID: {score:.4f}")

if __name__ == "__main__":
    main()
