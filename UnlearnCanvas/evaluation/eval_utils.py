#!/usr/bin/env python3
"""
Common utilities for evaluation scripts.
"""
import torch
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

# Import from retrack for consistent LoRA handling
sys.path.insert(0, str(Path(__file__).parent.parent / "machine_unlearning" / "retrack_latent" / "src"))
from utils import setup_lora_unet


@torch.no_grad()
def load_lora_into_unet(unet: UNet2DConditionModel, lora_path: str) -> UNet2DConditionModel:
    """
    Load LoRA weights into an already LoRA-enabled UNet in-place.
    
    This is used to swap LoRA weights without reloading the base UNet,
    significantly reducing loading time when processing multiple LoRA checkpoints.
    
    Args:
        unet: UNet with LoRA structure already applied (via setup_lora_unet)
        lora_path: Path to LoRA safetensors file
        
    Returns:
        unet: Same UNet instance with updated LoRA weights
    """
    state = load_file(lora_path)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    # Note: missing keys are expected (base UNet params), unexpected should be empty
    if unexpected:
        print(f"Warning: Unexpected keys when loading LoRA: {unexpected[:5]}...")
    return unet


def load_sd15_with_lora(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: Optional[str] = None,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    device: str = "cuda"
) -> Tuple[UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTokenizer, DDPMScheduler]:
    """
    Load SD1.5 components with optional LoRA weights.
    Uses the same LoRA loading method as retrack_latent for consistency.
    
    Args:
        model_id: HuggingFace model ID
        lora_path: Path to LoRA safetensors file
        lora_rank: LoRA rank (must match training)
        lora_alpha: LoRA alpha (must match training)
        device: Device to load models on
        
    Returns:
        unet, vae, text_encoder, tokenizer, scheduler
    """
    print(f"Loading SD1.5 from {model_id}...")
    
    # Load base components
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Load LoRA if provided (using retrack method)
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}...")
        # Apply LoRA structure
        unet = setup_lora_unet(unet, rank=lora_rank, alpha=lora_alpha)
        # Load LoRA weights
        lora_state_dict = load_file(lora_path)
        unet.load_state_dict(lora_state_dict, strict=False)
    
    unet = unet.to(device)
    
    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    return unet, vae, text_encoder, tokenizer, scheduler


@torch.no_grad()
def encode_prompt(
    prompt: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode text prompt to embeddings.
    
    Args:
        prompt: Text prompt
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        device: Device
        
    Returns:
        prompt_embeds: (1, 77, 768) text embeddings
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    prompt_embeds = text_encoder(text_input_ids)[0]
    return prompt_embeds


@torch.no_grad()
def generate_images_from_prompts(
    prompts: List[str],
    pipeline: StableDiffusionPipeline,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> List[Image.Image]:
    """
    Generate images from text prompts.
    
    Args:
        prompts: List of text prompts
        pipeline: Stable Diffusion pipeline
        num_images_per_prompt: Number of images per prompt
        guidance_scale: CFG scale (7.5 is standard)
        num_inference_steps: Number of denoising steps (50 is standard)
        seed: Random seed for reproducibility
        output_dir: Optional directory to save images
        
    Returns:
        List of PIL images
    """
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None
    
    all_images = []
    
    for idx, prompt in enumerate(prompts):
        images = pipeline(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images
        
        all_images.extend(images)
        
        # Save if output_dir specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for img_idx, img in enumerate(images):
                img_path = output_dir / f"image_{idx:04d}_{img_idx:02d}.png"
                img.save(img_path)
    
    return all_images


def load_prompts_from_jsonl(
    prompt_file: Path,
    style_filter: Optional[str] = None,
    prompt_type: str = "train",  # train, eval_weak, eval_medium, eval_strong
    max_prompts: Optional[int] = None
) -> List[Dict]:
    """
    Load prompts from JSONL file.
    
    Args:
        prompt_file: Path to train_prompts.jsonl
        style_filter: Optional style name to filter by
        prompt_type: Type of prompt to use
        max_prompts: Maximum number of prompts to load
        
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    
    with open(prompt_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Filter by style if specified
            if style_filter and data.get('style_name') != style_filter:
                continue
            
            # Get appropriate prompt
            if prompt_type == "train":
                prompt_text = data.get('training_prompt', '')
            elif prompt_type.startswith("eval_"):
                strength = prompt_type.split('_')[1]  # weak, medium, strong
                prompt_text = data.get('evaluation_prompts', {}).get(strength, '')
            else:
                prompt_text = data.get('training_prompt', '')
            
            if prompt_text:
                prompts.append({
                    'prompt': prompt_text,
                    'style': data.get('style_name'),
                    'image_id': data.get('image_id'),
                    'content': data.get('content_description')
                })
            
            if max_prompts and len(prompts) >= max_prompts:
                break
    
    return prompts


def get_all_styles(data_root: Path) -> List[str]:
    """
    Get list of all style names from data directory.
    
    Args:
        data_root: Path to UnlearnCanvas data directory
        
    Returns:
        Sorted list of style names
    """
    styles = []
    for path in data_root.iterdir():
        if path.is_dir() and path.name != "Seed_Images" and not path.name.startswith('.'):
            styles.append(path.name)
    return sorted(styles)


def load_style_images(
    data_root: Path,
    style_name: str,
    max_images: Optional[int] = None
) -> List[Path]:
    """
    Get paths to all images for a specific style.
    
    Args:
        data_root: Path to UnlearnCanvas data directory
        style_name: Name of the style
        max_images: Maximum number of images to return
        
    Returns:
        List of image paths
    """
    style_dir = data_root / style_name
    if not style_dir.exists():
        raise ValueError(f"Style directory not found: {style_dir}")
    
    # Search recursively for images (styles have subdirectories for subjects)
    image_paths = sorted(style_dir.glob("**/*.jpg")) + sorted(style_dir.glob("**/*.png"))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    return image_paths

