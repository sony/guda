"""
Generate sample images using trained LoRA models.
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import setup_lora_unet, load_lora_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples with trained LoRA')
    
    parser.add_argument('--lora_path', type=str, required=True,
                       help='Path to LoRA weights (.safetensors)')
    parser.add_argument('--prompts', type=str, nargs='+', required=True,
                       help='List of prompts to generate')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated images')
    parser.add_argument('--model_id', type=str, 
                       default='runwayml/stable-diffusion-v1-5',
                       help='Base model ID')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for models')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--num_images_per_prompt', type=int, default=4,
                       help='Number of images to generate per prompt')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Generating Samples with LoRA")
    print("=" * 80)
    print(f"LoRA path: {args.lora_path}")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print(f"Prompts: {len(args.prompts)}")
    print(f"Images per prompt: {args.num_images_per_prompt}")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load base pipeline components individually to avoid offload_state_dict issue
    print("Loading base model components...")
    
    # Load text encoder and tokenizer
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer",
        cache_dir=args.cache_dir
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    )
    
    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler",
        cache_dir=args.cache_dir
    )
    
    # Apply LoRA structure
    print(f"Applying LoRA structure (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    unet = setup_lora_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    unet = load_lora_weights(unet, args.lora_path)
    
    # Create pipeline from components
    print("Creating pipeline...")
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    
    # Move to device
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    
    print("\nGenerating images...")
    print("-" * 80)
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(args.prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(args.prompts)}: {prompt}")
        
        # Generate multiple images
        images = pipe(
            prompt=[prompt] * args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=args.device).manual_seed(args.seed + prompt_idx)
        ).images
        
        # Save images
        for img_idx, image in enumerate(images):
            filename = f"prompt{prompt_idx:02d}_img{img_idx:02d}.png"
            filepath = os.path.join(args.output_dir, filename)
            image.save(filepath)
            print(f"  Saved: {filename}")
    
    print("\n" + "=" * 80)
    print(f"Generation complete! Images saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
