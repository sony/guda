#!/usr/bin/env python3
"""
Generate evaluation images for all styles using all-class model.

This script generates images using weak/medium/strong prompts for evaluation.
Images are used as reference for computing ELBO-based attributions.
"""
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.eval_utils import (
    load_prompts_from_jsonl,
    get_all_styles,
    generate_images_from_prompts
)

# Import from retrack for LoRA handling
sys.path.insert(0, str(Path(__file__).parent.parent / "machine_unlearning" / "retrack_latent" / "src"))
from utils import setup_lora_unet


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation images")
    parser.add_argument(
        "--allclass_lora",
        type=str,
        default="outputs/allclass_sd15_lr1e4/lora.safetensors",
        help="Path to all-class LoRA checkpoint"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/UnlearnCanvas",
        help="Path to UnlearnCanvas data directory"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="sd_uc_qwen_prompts/data_production/prompts/eval_prompts.jsonl",
        help="Path to eval_prompts.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num_images_per_style",
        type=int,
        default=128,
        help="Number of images to generate per style per prompt type"
    )
    parser.add_argument(
        "--prompt_types",
        nargs='+',
        default=['eval_weak', 'eval_medium', 'eval_strong'],
        help="Prompt types to use"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--styles",
        nargs='+',
        default=None,
        help="Specific styles to generate (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    prompt_file = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get styles to process
    if args.styles:
        styles = args.styles
    else:
        styles = get_all_styles(data_root)
    
    print(f"Generating images for {len(styles)} styles")
    print(f"Prompt types: {args.prompt_types}")
    print(f"Images per style per prompt type: {args.num_images_per_style}")
    
    # Load pipeline with all-class LoRA
    print(f"\nLoading SD1.5 with all-class LoRA: {args.allclass_lora}")
    
    # Load components individually to avoid version compatibility issues
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Setup LoRA on UNet
    unet = setup_lora_unet(unet, rank=16, alpha=16)
    
    # Load LoRA weights
    lora_state_dict = load_file(args.allclass_lora)
    unet.load_state_dict(lora_state_dict, strict=False)
    
    # Create pipeline from components
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    
    pipeline = pipeline.to("cuda", dtype=torch.float16)
    pipeline.unet.eval()
    
    print("Pipeline loaded successfully\n")
    
    # Process each style
    for style_idx, style in enumerate(styles):
        print(f"[{style_idx+1}/{len(styles)}] Processing style: {style}")
        
        for prompt_type in args.prompt_types:
            print(f"  Prompt type: {prompt_type}")
            
            # Map prompt_type to intensity
            intensity_map = {
                'eval_weak': 'weak',
                'eval_medium': 'medium',
                'eval_strong': 'strong',
                'eval_ffsd': 'ffsd'
            }
            intensity = intensity_map.get(prompt_type, 'weak')
            
            # Load prompts for this style and intensity from eval_prompts.jsonl
            prompts = []
            with open(prompt_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Check if this prompt matches our style and intensity
                    if style in data['rel_path'] and data['intensity'] == intensity:
                        prompts.append(data['prompt'])
                        if len(prompts) >= args.num_images_per_style:
                            break
            
            if len(prompts) < args.num_images_per_style:
                print(f"    Warning: Only {len(prompts)} prompts available (requested {args.num_images_per_style})")
            
            # Output directory for this style and prompt type
            style_output_dir = output_dir / style / prompt_type
            style_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate images
            print(f"    Generating {len(prompts)} images...")
            images = generate_images_from_prompts(
                prompts=prompts,
                pipeline=pipeline,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                output_dir=None  # We'll save manually
            )
            
            # Save images
            for img_idx, img in enumerate(images):
                img_path = style_output_dir / f"image_{img_idx:04d}.png"
                img.save(img_path)
            
            print(f"    Saved {len(images)} images to {style_output_dir}")
        
        print()
    
    print("Image generation complete!")
    print(f"Total images generated: {len(styles)} styles × {len(args.prompt_types)} prompt types × {args.num_images_per_style} images")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
