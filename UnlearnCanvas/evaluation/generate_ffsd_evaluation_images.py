#!/usr/bin/env python3
"""
Generate evaluation images for FFSD models using the All-Class model.

Generates 320 images (16 styles x 20 objects) using FFSD prompts.
Output structure: {output_dir}/{style}/{style}_{object}.png
"""
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import sys
import numpy as np

# Patch CLIPTextModel to handle offload_state_dict argument
_ORIG_CLIPTEXT_INIT = CLIPTextModel.__init__

def _patched_cliptext_init(self, config, *model_args, **model_kwargs):
    model_kwargs.pop("offload_state_dict", None)
    return _ORIG_CLIPTEXT_INIT(self, config, *model_args, **model_kwargs)

CLIPTextModel.__init__ = _patched_cliptext_init

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FFSD evaluation images")
    parser.add_argument(
        "--allclass_dir",
        type=str,
        default="outputs/allclass_sd15_ffsd_prompts_full",
        help="Directory containing all-class checkpoint"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=10000,
        help="Checkpoint step to use"
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default="data/UnlearnCanvas/eval_prompts_ffsd_very_relaxed.jsonl",
        help="Path to evaluation prompts file (NOT training prompts)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation_images_ffsd",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--styles",
        nargs='+',
        default=[
            "Abstractionism", "Artist Sketch", "Blossom Season", "Blue Blooming", 
            "Bricks", "Byzantine", "Cartoon", "Cold Warm", 
            "Color Fantasy", "Comic Etch", "Crayon", "Crypto Punks", 
            "Cubism", "Dadaism", "Dapple", "Defoliation"
        ],
        help="Specific styles to generate (default: 16 styles A-D)"
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
        help="Random seed"
    )
    return parser.parse_args()

def init_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda", cache_dir: str = None) -> StableDiffusionPipeline:
    """Initialize Stable Diffusion pipeline with base components."""
    print(f"Initializing pipeline with cache_dir: {cache_dir}")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", cache_dir=cache_dir)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", cache_dir=cache_dir)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)

    pipe = StableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device, dtype=torch.float16)
    pipe.unet.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_unet_weights(pipe: StableDiffusionPipeline, checkpoint_path: Path, device: str = "cuda") -> None:
    """Load UNet weights from checkpoint into existing pipeline."""
    if checkpoint_path.is_dir():
        # Load from Diffusers directory
        print(f"Loading UNet from Diffusers directory: {checkpoint_path}")
        # Check if 'unet' subfolder exists, otherwise assume root is unet
        subfolder = "unet" if (checkpoint_path / "unet").exists() else None
        
        loaded_unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=torch.float16
        )
        pipe.unet.load_state_dict(loaded_unet.state_dict())
        del loaded_unet
    else:
        # Load from .pt file
        print(f"Loading UNet from .pt file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract UNet state dict
        state_dict_key = "unet_state_dict"
        if state_dict_key not in checkpoint:
            # Try alternative keys
            if "unet" in checkpoint:
                state_dict_key = "unet"
            elif "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model_state_dict" in checkpoint:
                state_dict_key = "model_state_dict"
            else:
                # Assume the checkpoint itself is the state dict
                state_dict_key = None
        
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
            
        pipe.unet.load_state_dict(state_dict, strict=True)
    
    pipe.unet.to(device, dtype=torch.float16)

def load_eval_prompts(eval_prompts_file, target_styles):
    """
    Load evaluation prompts from dedicated eval prompts file.
    
    Args:
        eval_prompts_file: Path to eval_prompts_ffsd_very_relaxed.jsonl
        target_styles: List of styles to generate images for
        
    Returns:
        dict {style: [(object, prompt), ...]}
    """
    style_prompts = {style: [] for style in target_styles}
    
    print(f"Loading evaluation prompts from {eval_prompts_file}...")
    with open(eval_prompts_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                style = data['style']
                obj = data['object']
                prompt = data['prompt']
                
                # Handle style name variations (spaces vs underscores)
                style_normalized = style
                if style not in style_prompts:
                    if style.replace(' ', '_') in style_prompts:
                        style_normalized = style.replace(' ', '_')
                    elif style.replace('_', ' ') in style_prompts:
                        style_normalized = style.replace('_', ' ')
                    else:
                        continue  # Style not in target list
                
                # Store prompt
                style_prompts[style_normalized].append((obj, prompt))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line due to error: {e}")
                continue
    
    # Filter out empty styles
    final_prompts = {s: p for s, p in style_prompts.items() if p}
    
    print(f"Loaded prompts for {len(final_prompts)} styles")
    for style, prompts in list(final_prompts.items())[:3]:
        print(f"  {style}: {len(prompts)} objects")
        
    return final_prompts

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation prompts
    style_prompts = load_eval_prompts(args.eval_prompts_file, args.styles)
    
    total_images = sum(len(p) for p in style_prompts.values())
    print(f"Plan to generate {total_images} images across {len(style_prompts)} styles")
    
    # Load Pipeline
    print("Loading SD1.5 pipeline...")
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = init_pipeline(model_id, device, cache_dir=args.cache_dir)
    
    # Find checkpoint
    allclass_dir = Path(args.allclass_dir)
    checkpoint_path = allclass_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint_path.exists():
        checkpoint_path = allclass_dir / f"checkpoint_step_{args.checkpoint_step}.pt"
    if not checkpoint_path.exists():
        # Fallback to dir itself
        if (allclass_dir / "unet").exists():
            checkpoint_path = allclass_dir
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, trying {allclass_dir}")
            checkpoint_path = allclass_dir

    print(f"Loading UNet from {checkpoint_path}")
    load_unet_weights(pipeline, checkpoint_path, device)
    
    # Generate images
    print("Starting generation...")
    for style, prompts in style_prompts.items():
        print(f"Generating {len(prompts)} images for {style}...")
        style_dir = output_dir / style
        style_dir.mkdir(exist_ok=True)
        
        for obj, prompt in tqdm(prompts, desc=style):
            # Filename: {style}_{object}.png
            # Sanitize filename
            safe_obj = obj.replace(' ', '_').replace('/', '-')
            safe_style = style.replace(' ', '_')
            filename = f"{safe_style}_{safe_obj}.png"
            filepath = style_dir / filename
            
            # Always overwrite to ensure we get new images
            if filepath.exists():
                filepath.unlink()
                
            # Generate
            generator = torch.Generator(device).manual_seed(args.seed)
            image = pipeline(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
            image.save(filepath)
            
    print(f"Generation complete! Saved to {output_dir}")

if __name__ == "__main__":
    main()
