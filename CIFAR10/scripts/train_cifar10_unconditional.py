#!/usr/bin/env python3
# train_cifar10_unconditional.py

import argparse
from cifar10_diffusion.trainer import UnconditionalTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Train unconditional DDPM on CIFAR-10 with improvements")
    p.add_argument("--data_dir",       type=str,   default="data")
    p.add_argument("--dataset",         type=str,   default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use for training.")
    p.add_argument("--output_dir",     type=str,   default="checkpoints/cifar-uncond-improved")
    p.add_argument("--epochs",         type=int,   default=2600)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--timesteps",      type=int,   default=4000)
    p.add_argument("--beta_schedule",  type=str,   default="squaredcos_cap_v2")
    p.add_argument("--learn_sigma",    action="store_true", help="Learn the variance of the reverse process.")
    p.add_argument("--lambda_vlb",     type=float, default=0.001, help="Weight for the VLB term in hybrid objective.")
    p.add_argument("--ema_decay",      type=float, default=0.9999)
    p.add_argument("--ema_start",      type=int,   default=0)
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  type=str,   default="diffusion-improved")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--save_every", type=int, default=100, help="Frequency (in epochs) of saving permanent milestone checkpoints.")
    p.add_argument("--save_latest_every", type=int, default=100, help="Frequency (in epochs) of saving the 'latest' resume checkpoint.") # Add this line
    p.add_argument("--sample_every",   type=int,   default=20)
    p.add_argument("--exclude_classes",  type=str,   nargs="+", default=None, help="List of class indices to exclude (e.g., 1 2 3), or 'random' to exclude one random class each epoch.")
    p.add_argument("--group_manifest", type=str, default=None, help="Optional train manifest whose group_ids define exclusion groups.")
    return p.parse_args()


def main():
    """
    Parses arguments and starts the training process.
    """
    args = parse_args()
    args.wandb_project = f"{args.dataset}-{args.wandb_project}"
    trainer = UnconditionalTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
