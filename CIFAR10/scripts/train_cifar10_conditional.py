#!/usr/bin/env python3
# train_cifar10_conditional.py
import argparse
from cifar10_diffusion.trainer import ConditionalTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Train class-conditional DDPM on CIFAR-10")
    p.add_argument("--data_dir",       type=str,   default="data")
    p.add_argument("--dataset",         type=str,   default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use for training.")
    p.add_argument("--output_dir",     type=str,   default="checkpoints/cifar-cond-improved")
    p.add_argument("--epochs",         type=int,   default=400)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--timesteps",      type=int,   default=4000)
    p.add_argument("--beta_schedule",  type=str,   default="squaredcos_cap_v2")
    p.add_argument("--learn_sigma",    action="store_true")
    p.add_argument("--lambda_vlb",     type=float, default=0.001)
    p.add_argument("--ema_decay",      type=float, default=0.9999)
    p.add_argument("--ema_start",      type=int,   default=0)
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  type=str,   default="diffusion-improved")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--save_every", type=int, default=100, help="Frequency (in epochs) of saving permanent milestone checkpoints.")
    p.add_argument("--save_latest_every", type=int, default=100, help="Frequency (in epochs) of saving the 'latest' resume checkpoint.") # Add this line
    p.add_argument("--sample_every",   type=int,   default=5)
    p.add_argument("--drop_prob",      type=float, default=0.1, help="Probability for classifier-free guidance.")
    return p.parse_args()

def main():
    args = parse_args()
    args.wandb_project = f"{args.dataset}-{args.wandb_project}"
    trainer = ConditionalTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()