#!/usr/bin/env python3
"""
Compute UNA (Unlearning-based Attribution) scores for FFSD models using ReTrack.

This script follows the same batched processing approach as compute_logoa_ffsd.py.
Adapted for FFSD prompt format: "{object}, artistic style featuring {descriptors}"

UNA(style_i, image) = ELBO(M_all-class, image) - ELBO(M_unlearn-i, image)
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
    parser = argparse.ArgumentParser(description="Compute UNA scores for FFSD models")
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
        help="All-class checkpoint step to use"
    )
    parser.add_argument(
        "--retrack_dir",
        type=str,
        default="outputs/retrack_sd15_style_removed_lr1em6_lam2_0",
        help="Directory containing ReTrack checkpoint subdirectories"
    )
    parser.add_argument(
        "--retrack_checkpoint_step",
        type=int,
        default=5000,
        help="ReTrack checkpoint step to use"
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
        default="outputs/una_retrack_scores_16styles_step5000",
        help="Output directory for score files"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="una_retrack_scores.csv",
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


def load_unet_from_checkpoint(checkpoint_path: str, model_id: str, precision: str = "bf16"):
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_obj = Path(checkpoint_path)
    
    if path_obj.is_dir():
        subfolder = "unet" if (path_obj / "unet").exists() else None
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=dtype
        ).to(device)
        unet.eval()
        return unet
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtype
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'unet_state_dict' in checkpoint:
            state_dict = checkpoint['unet_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        unet.load_state_dict(state_dict, strict=True)
        unet.eval()
        return unet


def main():
    args = parse_args()
    
    # Setup paths
    allclass_dir = Path(args.allclass_dir)
    retrack_dir = Path(args.retrack_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation prompts
    eval_prompts = load_eval_prompts(args.eval_prompts_file)
    
    # Find all ReTrack styles
    if args.styles:
        all_retrack_styles = sorted(args.styles)
        for style in all_retrack_styles:
            style_dir = retrack_dir / style
            if not style_dir.is_dir():
                print(f"Warning: Style directory not found: {style_dir}")
    else:
        all_retrack_styles = sorted([
            d.name for d in retrack_dir.iterdir()
            if d.is_dir() and (
                (d / f"checkpoint_step_{args.retrack_checkpoint_step}.pt").exists() or
                (d / f"checkpoint_step_{args.retrack_checkpoint_step}").exists()
            )
        ])
    
    # Setup precision
    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_amp = True
    else:
        amp_dtype = torch.float32
        use_amp = False
    
    print(f"Computing UNA scores for FFSD models")
    print(f"=" * 80)
    print(f"All-class dir: {allclass_dir}")
    print(f"ReTrack dir: {retrack_dir}")
    print(f"Image dir: {image_dir}")
    print(f"ReTrack models found: {len(all_retrack_styles)}")
    print(f"=" * 80 + "\n")
    
    # Collect all images
    all_image_paths = []
    all_image_metadata = []
    
    for style_dir in sorted(image_dir.iterdir()):
        if not style_dir.is_dir():
            continue
        generated_style = style_dir.name
        # Convert style name to filename format (spaces to underscores)
        style_prefix = generated_style.replace(' ', '_') + '_'
        
        for img_path in sorted(style_dir.glob("*.png")):
            filename = img_path.stem
            
            # Remove style prefix from filename to get object name
            if filename.startswith(style_prefix):
                object_name = filename[len(style_prefix):]
            else:
                # Fallback: try to extract object from filename
                filename_parts = filename.split('_', 1)
                object_name = filename_parts[1] if len(filename_parts) >= 2 else filename
            
            all_image_paths.append(img_path)
            all_image_metadata.append({
                'generated_style': generated_style,
                'object_name': object_name,
                'rel_path': str(img_path.relative_to(image_dir))
            })
    
    num_images = len(all_image_paths)
    print(f"Total images to process: {num_images}")
    
    # Load shared components
    print("Loading shared components...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32).to(device)
    text_encoder.eval()
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Encode images
    print(f"Encoding {num_images} images...")
    all_latents = []
    for i in tqdm(range(0, num_images, args.batch_size), desc="Encoding"):
        batch_paths = all_image_paths[i:i + args.batch_size]
        images = []
        for img_path in batch_paths:
            image = Image.open(img_path).convert("RGB")
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            images.append(image)
        images = torch.stack(images).to(device)
        images = 2.0 * images - 1.0
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        all_latents.append(latents.cpu())
    all_latents = torch.cat(all_latents, dim=0).to(device)
    
    # Generate prompt embeddings
    print("Generating prompt embeddings...")
    all_prompt_embeds = []
    for i in tqdm(range(0, num_images, args.batch_size), desc="Encoding prompts"):
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
        text_input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            batch_embeds = text_encoder(text_input_ids)[0]
        all_prompt_embeds.append(batch_embeds)
    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    
    # Pre-generate noise
    print("Pre-generating noise...")
    torch.manual_seed(args.seed)
    timesteps = torch.arange(args.min_timestep, args.max_timestep + 1, args.timestep_stride, device=device)
    noise_dtype = amp_dtype if use_amp else torch.float32
    pre_noise = torch.randn(
        args.num_noise_samples, len(timesteps), num_images,
        all_latents.shape[1], all_latents.shape[2], all_latents.shape[3],
        device=device, dtype=noise_dtype
    )
    
    # Load all-class UNet and precompute ELBO
    print("Loading all-class UNet...")
    allclass_checkpoint_pt = allclass_dir / f"checkpoint_step_{args.checkpoint_step}.pt"
    allclass_checkpoint_dir = allclass_dir / f"checkpoint-{args.checkpoint_step}"
    if allclass_checkpoint_dir.exists():
        allclass_checkpoint = allclass_checkpoint_dir
    elif allclass_checkpoint_pt.exists():
        allclass_checkpoint = allclass_checkpoint_pt
    else:
        if (allclass_dir / "unet").exists():
             allclass_checkpoint = allclass_dir
        else:
             raise FileNotFoundError(f"All-class checkpoint not found for step {args.checkpoint_step}")

    unet_template = load_unet_from_checkpoint(str(allclass_checkpoint), model_id, precision=args.precision)
    
    print("Precomputing all-class ELBO...")
    allclass_elbo_cache = []
    for batch_idx in tqdm(range(0, num_images, args.batch_size), desc="All-class ELBO"):
        batch_end = min(batch_idx + args.batch_size, num_images)
        batch_latents = all_latents[batch_idx:batch_end]
        batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
        batch_noise = pre_noise[:, :, batch_idx:batch_end]
        
        elbo_allclass = precompute_elbo_for_latent(
            latent=batch_latents,
            prompt_embeds=batch_prompt_embeds,
            unet=unet_template,
            scheduler=scheduler,
            timesteps=timesteps,
            noise=batch_noise,
            num_samples=args.num_noise_samples,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )
        allclass_elbo_cache.append(elbo_allclass.cpu())
    allclass_elbo_cache = torch.cat(allclass_elbo_cache, dim=0)
    
    # Process ReTrack models
    output_csv = output_dir / args.output_filename
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['generated_style', 'object_name', 'attribution_style', 'una_score', 'image_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for retrack_idx, attribution_style in enumerate(all_retrack_styles):
            print(f"\n[{retrack_idx+1}/{len(all_retrack_styles)}] Processing: {attribution_style}")
            
            retrack_checkpoint_dir = retrack_dir / attribution_style / f"checkpoint_step_{args.retrack_checkpoint_step}"
            retrack_checkpoint_pt = retrack_dir / attribution_style / f"checkpoint_step_{args.retrack_checkpoint_step}.pt"
            
            if retrack_checkpoint_dir.exists():
                retrack_checkpoint = retrack_checkpoint_dir
            elif retrack_checkpoint_pt.exists():
                retrack_checkpoint = retrack_checkpoint_pt
            else:
                print(f"  Warning: Checkpoint not found for {attribution_style}, skipping")
                continue
            
            try:
                unet_retrack = load_unet_from_checkpoint(str(retrack_checkpoint), model_id, precision=args.precision)
                
                all_scores = []
                for batch_idx in tqdm(range(0, num_images, args.batch_size), desc=f"  {attribution_style}", leave=False):
                    batch_end = min(batch_idx + args.batch_size, num_images)
                    batch_latents = all_latents[batch_idx:batch_end]
                    batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
                    batch_noise = pre_noise[:, :, batch_idx:batch_end]
                    batch_allclass_elbo = allclass_elbo_cache[batch_idx:batch_end].to("cuda")
                    
                    delta_elbo = compute_delta_elbo_with_cache(
                        latent=batch_latents,
                        prompt_embeds=batch_prompt_embeds,
                        unet_base=unet_retrack,
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
                        'una_score': score.item(),
                        'image_path': metadata['rel_path']
                    })
                
                del unet_retrack
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error processing ReTrack model {attribution_style}: {e}")
                continue
    
    print(f"UNA computation complete! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
