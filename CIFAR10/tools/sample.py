import argparse
from cifar10_diffusion.sampler import UnconditionalSampler, ConditionalSampler

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained diffusion model.")
    
    # Common arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["unconditional", "conditional"], help="Type of model to sample from.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Dataset the model was trained on.")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save the generated images.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for sampling.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of steps for the scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm", help="Scheduler to use (ddpm or ddim).")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta for DDIM scheduler.")
    parser.add_argument("--learn_sigma", action="store_true", help="Set if the model was trained to learn variance.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model weights for sampling.")
    parser.add_argument("--ema_path", type=str, default=None, help="Path to EMA weights, if different from ckpt_dir.")
    parser.add_argument("--noise_file", type=str, default=None, help="Path to a fixed noise tensor for deterministic sampling.")

    # Conditional-only arguments
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Scale for classifier-free guidance.")
    parser.add_argument("--num_images_per_class", type=int, default=10, help="Number of images to generate per class.")
    
    # Unconditional-only arguments
    parser.add_argument("--num_images", type=int, default=64, help="Total number of images to generate.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.model_type == "unconditional":
        sampler = UnconditionalSampler(args)
    elif args.model_type == "conditional":
        sampler = ConditionalSampler(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
    sampler.sample()

if __name__ == "__main__":
    main()