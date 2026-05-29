#!/usr/bin/env python3
# train_cifar10_esd.py

import os
# Set environment variables to suppress warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE"] = "0"  # Disable torch.compile for better multi-GPU compatibility
os.environ["ACCELERATE_DISABLE_RICH"] = "1"  # Disable rich progress bars for better compatibility
os.environ["TORCH_INDUCTOR_USE_STRIDED_OUTPUTS"] = "0"  # Disable problematic optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async CUDA operations
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # Completely disable dynamo/compile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import argparse
from cifar10_diffusion.esd_trainer import ESDUnlearningTrainer

def parse_args():
    p = argparse.ArgumentParser(description="ESD (Erased Stable Diffusion) Unlearning for CIFAR-10/100 Diffusion Models")
    
    # Basic training parameters
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--output_dir", type=str, default="checkpoints/cifar-esd")
    p.add_argument("--epochs", type=int, default=50, help="Number of unlearning epochs")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--pair_batch_size", type=int, default=128, help="Batch size for paired retain/forget batches")
    p.add_argument("--lr", type=float, default=1e-4)
    
    # Optimizer parameters
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], 
                   help="Optimizer type (adam recommended for ESD)")
    p.add_argument("--adam_b1", type=float, default=0.9, help="Adam beta1 parameter")
    p.add_argument("--adam_b2", type=float, default=0.999, help="Adam beta2 parameter")
    p.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon parameter")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    
    # Learning rate scheduler parameters
    p.add_argument("--lr_scheduler", type=str, default="cosine", 
                   choices=["cosine", "linear", "constant"], 
                   help="Learning rate scheduler type")
    p.add_argument("--warmup_steps", type=int, default=500, 
                   help="Number of warmup steps for learning rate scheduler")
    
    # Diffusion model parameters
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear")
    
    # EMA parameters
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_start", type=int, default=0)
    
    # Unlearning-specific parameters
    p.add_argument("--unlearn_class", type=int, required=True, 
                   help="Class to unlearn (0-9 for CIFAR-10, 0-99 for CIFAR-100)")
    p.add_argument("--teacher_model_path", type=str, required=True,
                   help="Path to the unconditional teacher model checkpoint")
    p.add_argument("--conditional_model_path", type=str, required=True,
                   help="Path to the conditional model checkpoint for negative guidance")
    p.add_argument("--group_manifest", type=str, default=None,
                   help="Optional train manifest whose group_ids define forget/retain groups")
    
    # ESD parameters
    p.add_argument("--lambda_forget", type=float, default=1.0,
                   help="Weight for forget loss in L_retain + λ * L_forget")
    p.add_argument("--guidance_weight", type=float, default=7.5,
                   help="Guidance weight w for negative guidance: ε_uncond - w * (ε_cond - ε_uncond)")
    p.add_argument("--retain_loss_type", type=str, default="standard", choices=["standard", "distillation"],
                   help="Type of retention loss: 'standard' uses diffusion loss, 'distillation' uses teacher-student loss")
    
    # Logging and checkpointing
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cifar-esd")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")
    p.add_argument("--wandb_group", type=str, default=None, help="WandB group name for organizing related runs")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="WandB tags for this run")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--save_latest_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=25)
    
    # Reproducibility
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return p.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function to start ESD unlearning process.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Validate arguments
    if args.dataset == "cifar10" and (args.unlearn_class < 0 or args.unlearn_class > 9):
        raise ValueError("For CIFAR-10, unlearn_class must be between 0 and 9")
    elif args.dataset == "cifar100" and (args.unlearn_class < 0 or args.unlearn_class > 99):
        raise ValueError("For CIFAR-100, unlearn_class must be between 0 and 99")
    
    # Validate ESD parameters
    if args.lambda_forget <= 0:
        raise ValueError(f"lambda_forget must be > 0, got {args.lambda_forget}")
    
    if args.guidance_weight <= 0:
        raise ValueError(f"guidance_weight must be > 0, got {args.guidance_weight}")
    
    # Validate model paths
    if not os.path.exists(args.teacher_model_path):
        raise ValueError(f"Teacher model path does not exist: {args.teacher_model_path}")
    
    if not os.path.exists(args.conditional_model_path):
        raise ValueError(f"Conditional model path does not exist: {args.conditional_model_path}")
    
    # Force unconditional mode and disable learn_sigma
    args.conditional = False
    args.learn_sigma = False
    args.lambda_vlb = 0.0  # Not used in ESD
    
    # Update WandB project name
    args.wandb_project = f"{args.dataset}-{args.wandb_project}"
    
    # Print configuration
    print("=" * 70)
    print("ESD (Erasing Specific Discriminative) Unlearning Configuration")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Unlearn class: {args.unlearn_class}")
    print(f"Teacher model: {args.teacher_model_path}")
    print(f"Conditional model: {args.conditional_model_path}")
    if args.group_manifest:
        print(f"Group manifest: {args.group_manifest}")
    print(f"Lambda forget: {args.lambda_forget}")
    print(f"Guidance weight: {args.guidance_weight}")
    print(f"Retain loss type: {args.retain_loss_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Pair batch size: {args.pair_batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Beta schedule: {args.beta_schedule}")
    print(f"Random seed: {args.seed}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)
    
    # ESD method details
    print("ESD Method Details:")
    print("- Objective: L_retain + λ * L_forget")
    if args.retain_loss_type == 'distillation':
        print("- Retain loss: Distillation loss E[||ε_teacher(x_t) - ε_θ(x_t)||²]")
    else:
        print("- Retain loss: Standard diffusion loss E[||ε - ε_θ(x_t)||²]")
    print("- Forget loss: Negative guidance target E[||ε_neg_guided - ε_θ(x_t)||²]")
    print("- Negative guidance: ε_uncond - w * (ε_cond - ε_uncond)")
    print("- Requires conditional model for guidance calculation")
    print("- Unconditional model only (main model)")
    print("=" * 70)
    
    # Create and start trainer
    trainer = ESDUnlearningTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
