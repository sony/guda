#!/usr/bin/env python3
# compute_fid_cifar10_torchmetrics.py

import argparse
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm


class GeneratedImageDataset(Dataset):
    """
    Custom Dataset for loading generated images from a folder on-the-fly.
    """
    def __init__(self, img_dir, image_size=32):
        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, '*.png')) +
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.jpeg'))
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),                         # Convert to [0,1] float32
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        # No resizing here; rely on metric to handle resizing if needed
        tensor = self.transform(img)
        return tensor


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
        "--dataset_split", type=str, default="train",
        help="Which split to use: 'train' (50k) or 'test' (10k)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for processing images"
    )
    return parser.parse_args()


def load_cifar10(split="train", batch_size=64):
    """
    Load CIFAR-10 dataset and prepare DataLoader.
    """
    is_train = (split == "train")
    transform = transforms.Compose([
        transforms.ToTensor()                           # [0,1] float32
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=is_train, download=True,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def compute_fid(real_loader, gen_loader, device="cuda:0"):  # noqa: C901
    """
    Compute FID score using torchmetrics.FrechetInceptionDistance.
    """
    # Initialize FID metric: normalize=True expects [0,1] float32 inputs
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Process real images
    print("Processing real images...")
    for real_batch, _ in tqdm(real_loader, desc="Real batches"):
        real_batch = real_batch.to(device, dtype=torch.float32)
        fid.update(real_batch, real=True)

    # Process generated images
    print("Processing generated images...")
    for gen_batch in tqdm(gen_loader, desc="Generated batches"):
        gen_batch = gen_batch.to(device, dtype=torch.float32)
        fid.update(gen_batch, real=False)

    # Compute final FID score
    score = fid.compute().item()
    return score


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load real CIFAR-10 images
    real_loader = load_cifar10(
        split=args.dataset_split,
        batch_size=args.batch_size
    )

    # Determine folders to process
    if args.pattern:
        folders = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))
    else:
        folders = [args.base_dir]

    if not folders:
        print(f"No folders match {args.base_dir}/{args.pattern}")
        return

    for gen_dir in folders:
        name = os.path.basename(os.path.normpath(gen_dir))
        print(f"\n>>> Computing FID for {name} <<<")
        
        # Setup generated image DataLoader
        gen_dataset = GeneratedImageDataset(gen_dir)
        if len(gen_dataset) == 0:
            print(f"Warning: No valid images found in {gen_dir}")
            continue

        gen_loader = DataLoader(
            gen_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"Found {len(gen_dataset)} images in {gen_dir}")
        score = compute_fid(real_loader, gen_loader, device=device)
        print(f"{name} → FID: {score:.4f}")


if __name__ == "__main__":
    main()
