#!/usr/bin/env python3
"""
Compute LOGOA (LOGO Attribution) scores for FFSD models.

This script processes all images for each LOGO model in batches.
Adapted for FFSD prompt format: "{object}, artistic style featuring {descriptors}"

LOGOA(style_i, image) = ELBO(M_all-class, image) - ELBO(M_logo-i, image)
"""
import argparse
import torch
import csv
import json
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.elbo_utils import (
    precompute_elbo_for_latent,
    compute_delta_elbo_with_cache
)
from evaluation.timing_utils import (
    time_block,
    print_timing_report
)
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer


# Patch CLIPTextModel to handle offload_state_dict argument
_ORIG_CLIPTEXT_INIT = CLIPTextModel.__init__


def _patched_cliptext_init(self, config, *model_args, **model_kwargs):
    model_kwargs.pop("offload_state_dict", None)
    return _ORIG_CLIPTEXT_INIT(self, config, *model_args, **model_kwargs)


CLIPTextModel.__init__ = _patched_cliptext_init


def parse_args():
    parser = argparse.ArgumentParser(description="Compute LOGOA scores for FFSD models")
    parser.add_argument(
        "--allclass_dir",
        type=str,
        default="outputs/allclass_sd15_uc_prompts_full",
        help="Directory containing all-class checkpoint"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=7500,
        help="Checkpoint step to use"
    )
    parser.add_argument(
        "--logo_dir",
        type=str,
        default="outputs/logo_sd15_uc_prompts_full",
        help="Directory containing LOGO checkpoint subdirectories"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="outputs/logoa_evaluation_images_16styles_step7500",
        help="Directory containing generated evaluation images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/logoa_scores_16styles_step7500",
        help="Output directory for score files"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="logoa_scores.csv",
        help="Filename for the output CSV"
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default="data/UnlearnCanvas/eval_prompts_ffsd_very_relaxed.jsonl",
        help="Path to evaluation prompts file (NOT training prompts)"
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
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=10,
        help="Minimum timestep for ELBO computation"
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=999,
        help="Maximum timestep for ELBO computation"
    )
    parser.add_argument(
        "--timestep_stride",
        type=int,
        default=10,
        help="Stride for timestep sampling"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision for UNet forward pass"
    )
    parser.add_argument(
        "--styles",
        nargs='+',
        default=None,
        help="Specific styles to compute attribution for"
    )
    return parser.parse_args()


def load_eval_prompts(eval_prompts_file: str) -> Dict[Tuple[str, str], str]:
    """
    Load evaluation prompts from the dedicated eval prompts file.
    
    Args:
        eval_prompts_file: Path to eval_prompts_ffsd_very_relaxed.jsonl
        
    Returns:
        Dict mapping (style, object) -> prompt
    """
    eval_prompts = {}
    print(f"Loading evaluation prompts from {eval_prompts_file}...")
    
    try:
        with open(eval_prompts_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    style = data['style']
                    obj = data['object']
                    prompt = data['prompt']
                    
                    # Store with both space and underscore variants as keys
                    eval_prompts[(style, obj)] = prompt
                    eval_prompts[(style.replace(' ', '_'), obj)] = prompt
                    eval_prompts[(style, obj.replace(' ', '_'))] = prompt
                    eval_prompts[(style.replace(' ', '_'), obj.replace(' ', '_'))] = prompt
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping line due to error: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading eval prompts file: {e}")
        raise
        
    print(f"Loaded {len(eval_prompts)} (style, object) -> prompt mappings")
    return eval_prompts


def find_checkpoint_file(checkpoint_dir: Path, checkpoint_step: int, raise_error: bool = True) -> Path:
    candidates = [
        checkpoint_dir / f"checkpoint-{checkpoint_step}",
        checkpoint_dir / f"checkpoint_step_{checkpoint_step}.pt",
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "model.safetensors"
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    if raise_error:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir} for step {checkpoint_step}. "
            f"Looked for: {[c.name for c in candidates]}"
        )
    return None


def init_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda") -> tuple:
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet_template = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder = text_encoder.to(device, dtype=torch.float16)
    vae = vae.to(device, dtype=torch.float16)
    
    text_encoder.eval()
    vae.eval()
    
    return tokenizer, text_encoder, vae, scheduler, unet_template


def load_unet_checkpoint(unet_template: UNet2DConditionModel, checkpoint_path: Path, device: str = "cuda") -> UNet2DConditionModel:
    from copy import deepcopy
    unet = deepcopy(unet_template)
    
    if checkpoint_path.is_dir():
        subfolder = "unet" if (checkpoint_path / "unet").exists() else None
        loaded_unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=torch.float16
        )
        unet.load_state_dict(loaded_unet.state_dict())
        del loaded_unet
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict_key = "unet_state_dict"
        if state_dict_key not in checkpoint:
            if "unet" in checkpoint:
                state_dict_key = "unet"
            elif "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            else:
                available = ", ".join(sorted(checkpoint.keys()))
                raise KeyError(f"No UNet weights in checkpoint {checkpoint_path}. Keys: {available}")

        unet.load_state_dict(checkpoint[state_dict_key], strict=True)
    
    unet = unet.to(device, dtype=torch.float16)
    unet = unet.to(memory_format=torch.channels_last).eval()
    
    return unet


@torch.no_grad()
def load_and_encode_images(image_paths: List[Path], vae, device: str = "cuda") -> torch.Tensor:
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        images.append(image)
    
    images = torch.stack(images).to(device)
    images = 2.0 * images - 1.0
    images = images.half()
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    latents = latents.float()
    return latents


def collect_all_images(image_dir: Path) -> Tuple[List[Path], List[Dict]]:
    image_paths = []
    image_metadata = []
    
    for style_dir in sorted(image_dir.iterdir()):
        if not style_dir.is_dir():
            continue
        
        style_name = style_dir.name
        # Convert style name to filename format (spaces to underscores)
        style_prefix = style_name.replace(' ', '_') + '_'
        
        for img_path in sorted(style_dir.glob("*.png")):
            filename = img_path.stem
            
            # Remove style prefix from filename to get object name
            if filename.startswith(style_prefix):
                object_name = filename[len(style_prefix):]
            else:
                # Fallback: try to extract object from filename
                parts = filename.split('_', 1)
                object_name = parts[1] if len(parts) >= 2 else "unknown"
            
            image_paths.append(img_path)
            image_metadata.append({
                'generated_style': style_name,
                'object_name': object_name,
                'rel_path': str(img_path.relative_to(image_dir))
            })
    
    return image_paths, image_metadata


def main():
    args = parse_args()
    
    # Setup paths
    allclass_dir = Path(args.allclass_dir)
    logo_dir = Path(args.logo_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation prompts
    eval_prompts = load_eval_prompts(args.eval_prompts_file)
    
    # Get ALL LOGO styles
    if args.styles:
        all_logo_styles = sorted(args.styles)
        for style in all_logo_styles:
            style_dir = logo_dir / style
            if not style_dir.is_dir():
                print(f"Warning: Style directory not found: {style_dir}")
            elif find_checkpoint_file(style_dir, args.checkpoint_step, raise_error=False) is None:
                print(f"Warning: Checkpoint not found for style: {style}")
    else:
        all_logo_styles = sorted([
            d.name for d in logo_dir.iterdir() 
            if d.is_dir() and find_checkpoint_file(d, args.checkpoint_step, raise_error=False) is not None
        ])
    
    # Setup precision
    use_amp = args.precision in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    
    print(f"Computing LOGOA scores for FFSD models")
    print(f"=" * 80)
    print(f"All-class dir: {allclass_dir}")
    print(f"LOGO dir: {logo_dir}")
    print(f"Image dir: {image_dir}")
    print(f"Checkpoint step: {args.checkpoint_step}")
    print(f"LOGO models found: {len(all_logo_styles)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Precision: {args.precision}")
    print(f"=" * 80 + "\n")
    
    # Collect all images
    print("Collecting all images...")
    all_image_paths, all_image_metadata = collect_all_images(image_dir)
    total_images = len(all_image_paths)
    print(f"Total images to process: {total_images}")
    
    # Initialize base SD1.5 components
    print("Loading base SD1.5 components...")
    tokenizer, text_encoder, vae, scheduler, unet_template = init_pipeline(device="cuda")
    
    # Encode all images
    print(f"Encoding {total_images} images to latent space...")
    all_latents = []
    for i in tqdm(range(0, total_images, args.batch_size), desc="Encoding batches"):
        batch_paths = all_image_paths[i:i + args.batch_size]
        batch_latents = load_and_encode_images(batch_paths, vae, device="cuda")
        all_latents.append(batch_latents)
    all_latents = torch.cat(all_latents, dim=0)
    
    # Generate prompt embeddings
    print("Generating prompt embeddings for each image...")
    all_prompt_embeds = []
    
    for i in tqdm(range(0, total_images, args.batch_size), desc="Encoding prompts"):
        batch_metadata = all_image_metadata[i:i + args.batch_size]
        batch_prompts = []
        
        for meta in batch_metadata:
            style = meta['generated_style']
            obj = meta['object_name']
            
            # Get prompt directly from eval prompts
            prompt = eval_prompts.get((style, obj))
            if not prompt:
                # Fallback if prompt not found
                print(f"Warning: No prompt found for ('{style}', '{obj}'), using fallback prompt")
                prompt = f"An artwork of {obj}, with style features"
            
            batch_prompts.append(prompt)
        
        text_inputs = tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to("cuda")
        
        with torch.no_grad():
            batch_embeds = text_encoder(text_input_ids)[0]
        
        all_prompt_embeds.append(batch_embeds)
            
    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    
    # Pre-generate noise
    print(f"Pre-generating noise samples...")
    torch.manual_seed(args.seed)
    timesteps = torch.tensor(
        list(range(args.min_timestep, args.max_timestep + 1, args.timestep_stride)),
        device="cuda",
        dtype=torch.long
    )
    noise_dtype = amp_dtype if use_amp else torch.float32
    pre_noise = torch.randn(
        args.num_noise_samples,
        len(timesteps),
        total_images,
        all_latents.shape[1],
        all_latents.shape[2],
        all_latents.shape[3],
        device="cuda",
        dtype=noise_dtype
    )
    
    # Load all-class model and pre-compute ELBO
    print("Loading all-class model...")
    allclass_checkpoint = find_checkpoint_file(allclass_dir, args.checkpoint_step)
    unet_allclass = load_unet_checkpoint(unet_template, allclass_checkpoint, device="cuda")
    
    print("Pre-computing ELBO for all-class model...")
    allclass_elbo_cache = []
    for batch_idx in tqdm(range(0, total_images, args.batch_size), desc="Computing all-class ELBO"):
        batch_end = min(batch_idx + args.batch_size, total_images)
        batch_latents = all_latents[batch_idx:batch_end]
        batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
        batch_noise = pre_noise[:, :, batch_idx:batch_end]
        
        elbo_allclass = precompute_elbo_for_latent(
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
        allclass_elbo_cache.append(elbo_allclass.cpu())
    
    allclass_elbo_cache = torch.cat(allclass_elbo_cache, dim=0)
    del unet_allclass
    torch.cuda.empty_cache()
    
    # Process LOGO models
    output_csv = output_dir / args.output_filename
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['generated_style', 'object_name', 'attribution_style', 'logoa_score', 'image_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for logo_idx, attribution_style in enumerate(all_logo_styles):
            print(f"\n[{logo_idx+1}/{len(all_logo_styles)}] Processing LOGO model: {attribution_style}")
            logo_checkpoint_dir = logo_dir / attribution_style
            
            try:
                logo_checkpoint = find_checkpoint_file(logo_checkpoint_dir, args.checkpoint_step)
                unet_logo = load_unet_checkpoint(unet_template, logo_checkpoint, device="cuda")
                
                all_scores = []
                for batch_idx in tqdm(range(0, total_images, args.batch_size), desc="Processing batches", leave=False):
                    batch_end = min(batch_idx + args.batch_size, total_images)
                    batch_latents = all_latents[batch_idx:batch_end]
                    batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
                    batch_noise = pre_noise[:, :, batch_idx:batch_end]
                    batch_allclass_elbo = allclass_elbo_cache[batch_idx:batch_end].to("cuda")
                    
                    delta_elbo = compute_delta_elbo_with_cache(
                        latent=batch_latents,
                        prompt_embeds=batch_prompt_embeds,
                        unet_base=unet_logo,
                        scheduler=scheduler,
                        timesteps=timesteps,
                        pre_noise=batch_noise,
                        elbo_modified_cache=batch_allclass_elbo,
                        num_samples=args.num_noise_samples,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype
                    )
                    all_scores.append(delta_elbo.cpu())
                
                all_scores = torch.cat(all_scores, dim=0)
                
                for img_idx, (metadata, score) in enumerate(zip(all_image_metadata, all_scores)):
                    writer.writerow({
                        'generated_style': metadata['generated_style'],
                        'object_name': metadata['object_name'],
                        'attribution_style': attribution_style,
                        'logoa_score': score.item(),
                        'image_path': metadata['rel_path']
                    })
                
                del unet_logo
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error processing LOGO model {attribution_style}: {e}")
                continue
    
    print(f"LOGOA computation complete! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
