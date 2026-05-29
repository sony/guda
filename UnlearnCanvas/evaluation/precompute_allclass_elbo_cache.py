#!/usr/bin/env python3
"""
Precompute all-class ELBO for all evaluation images.

This script computes ELBO values for the all-class model on all evaluation images
and saves them to a cache file. The cache includes:
- ELBO values for each image
- Noise generation parameters (seed, shape, timesteps) for reproducibility
- Image metadata (style, object, paths)

The cache is used by train_retrack_with_una.py to compute ΔELBO during unlearning
without needing to reload the all-class model.

Key design decisions:
- Does NOT save pre-noise tensor (would be ~10GB)
- Instead saves noise generation parameters for reproducibility
- Relies on PyTorch deterministic RNG: same seed + shape → same noise
- Compatible with compute_logoa_ffsd.py noise generation
"""
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.elbo_utils import precompute_elbo_for_latent
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


# Patch CLIPTextModel for compatibility
_ORIG_CLIPTEXT_INIT = CLIPTextModel.__init__

def _patched_cliptext_init(self, config, *model_args, **model_kwargs):
    model_kwargs.pop("offload_state_dict", None)
    return _ORIG_CLIPTEXT_INIT(self, config, *model_args, **model_kwargs)

CLIPTextModel.__init__ = _patched_cliptext_init


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute all-class ELBO cache")
    parser.add_argument(
        "--allclass_checkpoint",
        type=str,
        required=True,
        help="Path to all-class checkpoint"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing evaluation images"
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        required=True,
        help="Path to evaluation prompts JSONL"
    )
    parser.add_argument(
        "--output_cache",
        type=str,
        required=True,
        help="Output path for cache file (.pt)"
    )
    parser.add_argument(
        "--num_noise_samples",
        type=int,
        default=1,
        help="Number of noise samples per image"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=10,
        help="Minimum timestep"
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=999,
        help="Maximum timestep"
    )
    parser.add_argument(
        "--timestep_stride",
        type=int,
        default=10,
        help="Timestep stride"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision for computation"
    )
    return parser.parse_args()


def load_eval_prompts(eval_prompts_file: str) -> Dict[Tuple[str, str], str]:
    """Load evaluation prompts from JSONL file."""
    eval_prompts = {}
    print(f"Loading evaluation prompts from {eval_prompts_file}...")
    
    with open(eval_prompts_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            style = data['style']
            obj = data['object']
            prompt = data['prompt']
            
            # Store with both space and underscore variants
            eval_prompts[(style, obj)] = prompt
            eval_prompts[(style.replace(' ', '_'), obj)] = prompt
            eval_prompts[(style, obj.replace(' ', '_'))] = prompt
            eval_prompts[(style.replace(' ', '_'), obj.replace(' ', '_'))] = prompt
    
    print(f"Loaded {len(eval_prompts)} prompt mappings")
    return eval_prompts


def collect_all_images(image_dir: Path) -> Tuple[List[Path], List[Dict]]:
    """Collect all evaluation images and their metadata."""
    image_paths = []
    image_metadata = []
    
    for style_dir in sorted(image_dir.iterdir()):
        if not style_dir.is_dir():
            continue
        
        style_name = style_dir.name
        style_prefix = style_name.replace(' ', '_') + '_'
        
        for img_path in sorted(style_dir.glob("*.png")):
            filename = img_path.stem
            
            if filename.startswith(style_prefix):
                object_name = filename[len(style_prefix):]
            else:
                parts = filename.split('_', 1)
                object_name = parts[1] if len(parts) >= 2 else "unknown"
            
            image_paths.append(img_path)
            image_metadata.append({
                'generated_style': style_name,
                'object_name': object_name,
                'rel_path': str(img_path.relative_to(image_dir))
            })
    
    return image_paths, image_metadata


def init_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    """Initialize SD1.5 pipeline components."""
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    text_encoder = text_encoder.to(device, dtype=torch.float16).eval()
    vae = vae.to(device, dtype=torch.float16).eval()
    
    return tokenizer, text_encoder, vae, scheduler


def load_unet_checkpoint(checkpoint_path: Path, model_id: str, device: str = "cuda"):
    """Load UNet from checkpoint."""
    path_obj = Path(checkpoint_path)
    
    if path_obj.is_dir():
        subfolder = "unet" if (path_obj / "unet").exists() else None
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=torch.float16
        ).to(device)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('unet_state_dict') or checkpoint.get('state_dict') or checkpoint
        unet.load_state_dict(state_dict, strict=True)
    
    unet = unet.to(memory_format=torch.channels_last).eval()
    return unet


@torch.no_grad()
def load_and_encode_images(image_paths: List[Path], vae, device: str = "cuda"):
    """Load and encode images to latent space."""
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        images.append(image)
    
    images = torch.stack(images).to(device)
    images = 2.0 * images - 1.0
    images = images.half()
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    return latents.float()


def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = Path(args.image_dir)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup precision
    use_amp = args.precision in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if use_amp else torch.float32
    
    print("=" * 80)
    print("Precomputing All-Class ELBO Cache")
    print("=" * 80)
    print(f"All-class checkpoint: {args.allclass_checkpoint}")
    print(f"Image directory: {image_dir}")
    print(f"Output cache: {output_cache}")
    print(f"Precision: {args.precision}")
    print(f"Seed: {args.seed}")
    print("=" * 80 + "\n")
    
    # Load evaluation prompts
    eval_prompts = load_eval_prompts(args.eval_prompts_file)
    
    # Collect all images
    print("Collecting images...")
    all_image_paths, all_image_metadata = collect_all_images(image_dir)
    total_images = len(all_image_paths)
    print(f"Total images: {total_images}\n")
    
    # Initialize pipeline
    print("Loading SD1.5 components...")
    tokenizer, text_encoder, vae, scheduler = init_pipeline(device=device)
    
    # Load all-class UNet
    print("Loading all-class UNet...")
    unet_allclass = load_unet_checkpoint(
        Path(args.allclass_checkpoint),
        model_id="runwayml/stable-diffusion-v1-5",
        device=device
    )
    
    # Encode all images
    print(f"Encoding {total_images} images to latent space...")
    all_latents = []
    for i in tqdm(range(0, total_images, args.batch_size), desc="Encoding"):
        batch_paths = all_image_paths[i:i + args.batch_size]
        batch_latents = load_and_encode_images(batch_paths, vae, device=device)
        all_latents.append(batch_latents)
    all_latents = torch.cat(all_latents, dim=0)
    
    # Generate prompt embeddings
    print("Generating prompt embeddings...")
    all_prompt_embeds = []
    
    for i in tqdm(range(0, total_images, args.batch_size), desc="Encoding prompts"):
        batch_metadata = all_image_metadata[i:i + args.batch_size]
        batch_prompts = []
        
        for meta in batch_metadata:
            style = meta['generated_style']
            obj = meta['object_name']
            prompt = eval_prompts.get((style, obj))
            if not prompt:
                print(f"Warning: No prompt for ('{style}', '{obj}')")
                prompt = f"An artwork of {obj}, with style features"
            batch_prompts.append(prompt)
        
        text_inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        batch_embeds = text_encoder(text_input_ids)[0]
        all_prompt_embeds.append(batch_embeds)
    
    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    
    # Pre-generate noise (CRITICAL: same method as compute_logoa_ffsd.py)
    print(f"Pre-generating noise with seed={args.seed}...")
    torch.manual_seed(args.seed)
    
    timesteps = torch.tensor(
        list(range(args.min_timestep, args.max_timestep + 1, args.timestep_stride)),
        device=device,
        dtype=torch.long
    )
    
    noise_dtype = amp_dtype if use_amp else torch.float32
    noise_shape = (
        args.num_noise_samples,
        len(timesteps),
        total_images,
        all_latents.shape[1],  # 4
        all_latents.shape[2],  # 64
        all_latents.shape[3],  # 64
    )
    
    print(f"Noise shape: {noise_shape}")
    pre_noise = torch.randn(*noise_shape, device=device, dtype=noise_dtype)
    
    # Compute all-class ELBO
    print(f"Computing ELBO for all {total_images} images...")
    allclass_elbo_cache = []
    
    for batch_idx in tqdm(range(0, total_images, args.batch_size), desc="Computing ELBO"):
        batch_end = min(batch_idx + args.batch_size, total_images)
        batch_latents = all_latents[batch_idx:batch_end]
        batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
        batch_noise = pre_noise[:, :, batch_idx:batch_end]
        
        elbo = precompute_elbo_for_latent(
            latent=batch_latents,
            prompt_embeds=batch_prompt_embeds,
            unet=unet_allclass,
            scheduler=scheduler,
            timesteps=timesteps,
            noise=batch_noise,
            num_samples=args.num_noise_samples,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )
        allclass_elbo_cache.append(elbo.cpu())
    
    allclass_elbo_cache = torch.cat(allclass_elbo_cache, dim=0)
    
    # Save cache
    print(f"\nSaving cache to {output_cache}...")
    cache_data = {
        'elbo_values': allclass_elbo_cache,
        'image_metadata': all_image_metadata,
        'noise_params': {
            'seed': args.seed,
            'num_noise_samples': args.num_noise_samples,
            'min_timestep': args.min_timestep,
            'max_timestep': args.max_timestep,
            'timestep_stride': args.timestep_stride,
            'noise_shape': list(noise_shape),
            'noise_dtype': str(noise_dtype),
            'total_images': total_images,
        },
        'model_info': {
            'checkpoint': str(args.allclass_checkpoint),
            'precision': args.precision,
        },
        'timesteps': timesteps.cpu(),
    }
    
    torch.save(cache_data, output_cache)
    
    print(f"\n✓ Cache saved successfully!")
    print(f"  - ELBO values: {allclass_elbo_cache.shape}")
    print(f"  - File size: {output_cache.stat().st_size / 1024**2:.2f} MB")
    print(f"  - Noise reproducibility: seed={args.seed}, shape={noise_shape}")


if __name__ == "__main__":
    main()
