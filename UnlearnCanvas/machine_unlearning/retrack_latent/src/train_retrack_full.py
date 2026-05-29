#!/usr/bin/env python3
"""
ReTrack Unlearning for Full Fine-tuning Models (UNA - Unlearning-based Attribution)

This script implements ε-redirect unlearning to approximate LOGOA scores.
Unlike LoRA version, this trains the full UNet to forget specific styles.

Key features:
- Full UNet fine-tuning (not LoRA)
- wandb logging for loss curves
- Periodic image generation for qualitative monitoring
- Checkpoint saving at multiple steps for correlation analysis
"""
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
from tqdm import tqdm
import wandb
import random
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import re


# Patch CLIPTextModel for compatibility
_ORIG_CLIPTEXT_INIT = CLIPTextModel.__init__


def _patched_cliptext_init(self, config, *model_args, **model_kwargs):
    model_kwargs.pop("offload_state_dict", None)
    return _ORIG_CLIPTEXT_INIT(self, config, *model_args, **model_kwargs)


CLIPTextModel.__init__ = _patched_cliptext_init


def remove_style_from_prompt(prompt: str) -> str:
    """
    Remove style information from UC prompt.
    
    UC prompts follow pattern: "a photo of a {object} in {style} style"
    This function removes "in {style} style" part.
    
    Examples:
        "a photo of a dog in Van Gogh style" -> "a photo of a dog"
        "a photo of a cat in Impressionism style" -> "a photo of a cat"
    
    Args:
        prompt: Original prompt with style
    
    Returns:
        Prompt with style removed
    """
    # Pattern: "in {anything} style" at the end
    pattern = r'\s+in\s+[^\s]+(?:\s+[^\s]+)*\s+style'
    prompt_no_style = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
    return prompt_no_style.strip()


class UnlearnDataset(Dataset):
    """Dataset for ReTrack unlearning with forget and retain sets."""
    
    def __init__(self, prompts_file: Path, forget_style: str, data_root: Path):
        """
        Args:
            prompts_file: JSONL file with prompts
            forget_style: Style to forget
            data_root: Root directory containing images
        """
        self.data_root = data_root
        self.forget_style = forget_style
        
        # Load prompts
        import json
        self.forget_samples = []
        self.retain_samples = []
        # Index retain samples by (object, image_id) for matching
        self.retain_by_object_id = {}
        
        with open(prompts_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                rel_path = data['rel_path']  # Use rel_path field with .jpg extension
                prompt = data['prompt']
                
                # Parse style, object, and ID from rel_path: "UnlearnCanvas/Style/Object/N.jpg"
                parts = rel_path.split('/')
                if len(parts) >= 4:
                    style = parts[1]
                    object_name = parts[2]
                    image_id = parts[3].replace('.jpg', '')  # Remove extension
                    image_path = data_root / rel_path
                    
                    if not image_path.exists():
                        continue
                    
                    sample = {
                        'image_path': str(image_path),
                        'prompt': prompt,
                        'style': style,
                        'object': object_name,
                        'image_id': image_id
                    }
                    
                    if style == forget_style:
                        self.forget_samples.append(sample)
                    else:
                        self.retain_samples.append(sample)
                        # Index by (object, image_id) for anchor matching
                        key = (object_name, image_id)
                        if key not in self.retain_by_object_id:
                            self.retain_by_object_id[key] = []
                        self.retain_by_object_id[key].append(len(self.retain_samples) - 1)
        
        print(f"Loaded dataset:")
        print(f"  Forget ({forget_style}): {len(self.forget_samples)} samples")
        print(f"  Retain (other styles): {len(self.retain_samples)} samples")
        print(f"  Retain indexed by (object, id): {len(self.retain_by_object_id)} unique combinations")
    
    def __len__(self):
        # Balanced sampling: return larger of the two sets
        return max(len(self.forget_samples), len(self.retain_samples))
    
    def __getitem__(self, idx):
        # Alternate between forget and retain samples
        if idx % 2 == 0:
            # Forget sample
            sample_idx = (idx // 2) % len(self.forget_samples)
            sample = self.forget_samples[sample_idx]
            is_forget = True
        else:
            # Retain sample
            sample_idx = (idx // 2) % len(self.retain_samples)
            sample = self.retain_samples[sample_idx]
            is_forget = False
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB').resize((512, 512))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = 2.0 * image - 1.0  # Normalize to [-1, 1]
        
        return {
            'image': image,
            'prompt': sample['prompt'],
            'is_forget': is_forget,
            'style': sample['style'],
            'object': sample['object'],
            'image_id': sample['image_id']
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    is_forget = torch.tensor([item['is_forget'] for item in batch], dtype=torch.bool)
    styles = [item['style'] for item in batch]
    objects = [item['object'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'prompts': prompts,
        'is_forget': is_forget,
        'styles': styles,
        'objects': objects,
        'image_ids': image_ids
    }


def load_base_model(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    """Load base SD1.5 components."""
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Move to device
    text_encoder = text_encoder.to(device, dtype=torch.float16).eval()
    vae = vae.to(device, dtype=torch.float16).eval()
    unet = unet.to(device, dtype=torch.float32)  # Keep UNet in FP32 for training
    
    return tokenizer, text_encoder, vae, unet, scheduler


def load_allclass_checkpoint(unet: UNet2DConditionModel, checkpoint_path: Path):
    """Load all-class checkpoint into UNet."""
    if checkpoint_path.is_dir():
        # Load from Diffusers directory
        print(f"Loading UNet from Diffusers directory: {checkpoint_path}")
        # Check if 'unet' subfolder exists, otherwise assume root is unet
        subfolder = "unet" if (checkpoint_path / "unet").exists() else None
        
        loaded_unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder,
            torch_dtype=torch.float32
        )
        unet.load_state_dict(loaded_unet.state_dict())
        del loaded_unet
        print(f"✓ Loaded all-class checkpoint from {checkpoint_path}")
    else:
        # Load from .pt file
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract UNet state dict
        if "unet_state_dict" in checkpoint:
            state_dict = checkpoint["unet_state_dict"]
        elif "unet" in checkpoint:
            state_dict = checkpoint["unet"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError(f"No UNet weights found in checkpoint. Keys: {checkpoint.keys()}")
        
        unet.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded all-class checkpoint from {checkpoint_path}")


def create_pretrained_unet_copy(unet: UNet2DConditionModel, device: str = "cuda"):
    """Create a frozen copy of UNet for preservation loss (pre-trained reference)."""
    import copy
    unet_pretrained = copy.deepcopy(unet)
    unet_pretrained.eval()
    unet_pretrained.requires_grad_(False)  # Freeze all parameters
    unet_pretrained = unet_pretrained.to(device, dtype=torch.float32)
    print("✓ Created frozen pre-trained UNet copy for preservation loss")
    return unet_pretrained


@torch.no_grad()
def encode_images(images, vae):
    """Encode images to latent space."""
    # Convert to FP16 for VAE
    images = images.half()
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    return latents.float()  # Return FP32 for training


@torch.no_grad()
def get_text_embeddings(prompts, tokenizer, text_encoder):
    """Get text embeddings for prompts."""
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]
    return embeddings.float()  # Convert FP16 to FP32 for UNet compatibility


def compute_retrack_loss(
    unet,
    unet_pretrained,
    scheduler,
    latents_forget,
    latents_retain,
    text_embeds_forget,
    text_embeds_retain,
    anchor_indices=None,
    text_embeds_anchor=None,
    epsilon: float = 0.1,
    lambda_stabilize: float = 1.0
):
    """
    Compute ReTrack ε-redirect loss with anchor matching.
    
    Loss = ||ε_θ(forget) - ε_θ₀(anchor)||^2 + λ * ||ε_θ(retain) - ε_θ₀(retain)||^2
    
    First term: Redirect forget to anchor (either matched retain or style-removed)
    Second term: Preserve retain representations (compare with pre-trained model)
    
    Args:
        anchor_indices: (Optional) List of indices for matched_retain strategy
        text_embeds_anchor: (Optional) Pre-computed embeddings for style_removed strategy
    """
    device = latents_forget.device
    batch_size_forget = latents_forget.shape[0]
    batch_size_retain = latents_retain.shape[0]
    
    # Sample timesteps for each batch
    timesteps_forget = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size_forget,), device=device)
    timesteps_retain = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size_retain,), device=device)
    
    # Add noise separately for forget and retain
    noise_forget = torch.randn_like(latents_forget)
    noise_retain = torch.randn_like(latents_retain)
    noisy_latents_forget = scheduler.add_noise(latents_forget, noise_forget, timesteps_forget)
    noisy_latents_retain = scheduler.add_noise(latents_retain, noise_retain, timesteps_retain)
    
    # Predict noise with fine-tuning UNet
    noise_pred_forget = unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states=text_embeds_forget).sample
    noise_pred_retain = unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states=text_embeds_retain).sample
    
    # ε-redirect loss: Push forget predictions towards anchor predictions
    # Anchor can be either matched retain samples or style-removed prompts
    with torch.no_grad():
        if anchor_indices is not None:
            # Strategy 1: matched_retain - use matched retain samples (same object & ID)
            # CRITICAL: Use SAME noise and timesteps as Forget for meaningful comparison
            # We want: Map(Forget_image + noise_A, t) -> Target(Retain_image + noise_A, t)
            latents_anchor = latents_retain[anchor_indices]
            text_embeds_anchor_used = text_embeds_retain[anchor_indices]
            
            # ★ Use SAME noise and timesteps as forget
            noise_anchor = noise_forget
            timesteps_anchor = timesteps_forget
            
            noisy_latents_anchor = scheduler.add_noise(latents_anchor, noise_anchor, timesteps_anchor)
        else:
            # Strategy 2: style_removed - use forget latents with style-removed prompts
            # CRITICAL: Use the SAME noisy latents as forget, only change the text prompt
            # This ensures we're comparing the same noisy input with different conditioning
            text_embeds_anchor_used = text_embeds_anchor  # Pre-computed style-removed embeddings
            timesteps_anchor = timesteps_forget
            noisy_latents_anchor = noisy_latents_forget  # ★ Same noisy latents as forget!
        
        # Anchor predictions from pre-trained model
        noise_pred_anchor = unet_pretrained(noisy_latents_anchor, timesteps_anchor, encoder_hidden_states=text_embeds_anchor_used).sample
    
    loss_redirect = F.mse_loss(noise_pred_forget, noise_pred_anchor)
    
    # Preservation loss: Keep retain predictions close to pre-trained model predictions
    with torch.no_grad():
        noise_pred_retain_pretrained = unet_pretrained(noisy_latents_retain, timesteps_retain, encoder_hidden_states=text_embeds_retain).sample
    loss_preserve = F.mse_loss(noise_pred_retain, noise_pred_retain_pretrained)
    
    # Total loss
    loss = loss_redirect + lambda_stabilize * loss_preserve
    
    return loss, loss_redirect, loss_preserve


@torch.no_grad()
def generate_sample_images(
    unet,
    vae,
    tokenizer,
    text_encoder,
    forget_style: str,
    output_dir: Path,
    step: int,
    num_samples: int = 4,
    seed: int = 42
):
    """Generate sample images for monitoring unlearning progress."""
    device = unet.device
    original_dtype = next(unet.parameters()).dtype
    
    # Convert UNet to FP16 for inference
    unet = unet.half()
    
    # Create pipeline for generation
    scheduler_gen = DPMSolverMultistepScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    
    pipe = StableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler_gen,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe.set_progress_bar_config(disable=True)
    
    # Test styles: forget style + 3 retain styles for comparison
    # Retain styles are selected from styles NOT in the first 16 alphabetically
    # (fixed set to avoid inconsistency across runs)
    FIXED_RETAIN_STYLES = ["Dreamweave", "Expressionism", "Fauvism"]
    test_styles = [forget_style] + FIXED_RETAIN_STYLES
    test_objects = ["dog", "cat", "flower"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    
    idx = 0
    for style in test_styles:
        for obj in test_objects:
            prompt = f"a photo of a {obj} in {style} style"
            generator = torch.Generator(device=device).manual_seed(seed + idx)
            image = pipe(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            # Save image with descriptive name
            image_path = output_dir / f"step{step:06d}_{style.replace(' ', '_')}_{obj}.png"
            image.save(image_path)
            images.append(image)
            idx += 1
    
    # Restore UNet to original dtype
    unet = unet.to(dtype=original_dtype)
    
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="ReTrack unlearning (full fine-tuning)")
    parser.add_argument(
        "--allclass_checkpoint",
        type=str,
        required=True,
        help="Path to all-class checkpoint"
    )
    parser.add_argument(
        "--forget_style",
        type=str,
        required=True,
        help="Style to forget (unlearn)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/UnlearnCanvas/train_prompts_uc.jsonl",
        help="JSONL file with training prompts"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing UnlearnCanvas images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and samples"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5000,
        help="Total number of training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for redirect loss"
    )
    parser.add_argument(
        "--lambda_stabilize",
        type=float,
        default=1.0,
        help="Weight for stabilization loss"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        nargs='+',
        default=[1000, 2000, 3000, 4000, 5000],
        help="Steps at which to save checkpoints"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=500,
        help="Generate sample images every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="unlearn-canvas-retrack",
        help="wandb project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="wandb run name (default: retrack_{forget_style}_{timestamp})"
    )
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default="matched_retain",
        choices=["matched_retain", "style_removed"],
        help="Anchor strategy: matched_retain (same object/ID, different style) or style_removed (remove style from prompt)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    prompts_file = Path(args.prompts_file)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    lr_str = f"{args.learning_rate:.0e}".replace(".", "_").replace("e-0", "em")
    lam_str = f"{args.lambda_stabilize:.1f}".replace(".", "_")
    eps_str = f"{args.epsilon:.2f}".replace(".", "_")
    wandb_name = args.wandb_name or f"retrack_{args.forget_style}_lr{lr_str}_lam{lam_str}_eps{eps_str}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        config=vars(args)
    )
    
    print(f"ReTrack Unlearning (Full Fine-tuning)")
    print(f"=" * 80)
    print(f"Forget style: {args.forget_style}")
    print(f"Anchor strategy: {args.anchor_strategy}")
    print(f"All-class checkpoint: {args.allclass_checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Lambda stabilize: {args.lambda_stabilize}")
    print(f"=" * 80 + "\n")
    
    # Load base model
    print("Loading base SD1.5 model...")
    tokenizer, text_encoder, vae, unet, scheduler = load_base_model(device="cuda")
    
    # Load all-class checkpoint
    print(f"Loading all-class checkpoint: {args.allclass_checkpoint}")
    load_allclass_checkpoint(unet, Path(args.allclass_checkpoint))
    
    # Create pre-trained UNet copy for preservation loss
    print("\nCreating pre-trained UNet copy...")
    unet_pretrained = create_pretrained_unet_copy(unet, device="cuda")
    
    # Setup dataset
    print("\nPreparing dataset...")
    dataset = UnlearnDataset(prompts_file, args.forget_style, data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\nStarting unlearning training...")
    unet.train()
    global_step = 0
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while global_step < args.num_steps:
        for batch in dataloader:
            if global_step >= args.num_steps:
                break
            
            # Separate forget and retain samples
            images = batch['images'].to("cuda")
            prompts = batch['prompts']
            is_forget = batch['is_forget']
            objects = batch['objects']
            image_ids = batch['image_ids']
            
            # Split into forget and retain
            forget_mask = is_forget
            retain_mask = ~is_forget
            
            if forget_mask.sum() == 0 or retain_mask.sum() == 0:
                # Skip if batch doesn't have both forget and retain samples
                continue
            
            images_forget = images[forget_mask]
            images_retain = images[retain_mask]
            prompts_forget = [p for p, f in zip(prompts, is_forget) if f]
            prompts_retain = [p for p, f in zip(prompts, is_forget) if not f]
            objects_forget = [o for o, f in zip(objects, is_forget) if f]
            objects_retain = [o for o, f in zip(objects, is_forget) if not f]
            ids_forget = [i for i, f in zip(image_ids, is_forget) if f]
            ids_retain = [i for i, f in zip(image_ids, is_forget) if not f]
            
            # Prepare anchor based on strategy
            if args.anchor_strategy == "matched_retain":
                # Strategy 1: Match each forget sample to a retain sample with same object & ID
                anchor_indices = []
                for obj_f, id_f in zip(objects_forget, ids_forget):
                    # Find matching retain samples (same object & ID, different style)
                    matches = [i for i, (obj_r, id_r) in enumerate(zip(objects_retain, ids_retain))
                              if obj_r == obj_f and id_r == id_f]
                    
                    if matches:
                        # Randomly select one matching retain sample
                        anchor_idx = random.choice(matches)
                    else:
                        # Fallback: use random retain sample if no match
                        anchor_idx = random.randint(0, len(objects_retain) - 1)
                    
                    anchor_indices.append(anchor_idx)
                
                text_embeds_anchor = None
            else:
                # Strategy 2: style_removed - remove style from forget prompts
                prompts_forget_no_style = [remove_style_from_prompt(p) for p in prompts_forget]
                anchor_indices = None
                # Pre-compute style-removed embeddings
                with torch.no_grad():
                    text_embeds_anchor = get_text_embeddings(prompts_forget_no_style, tokenizer, text_encoder)
            
            # Encode images
            with torch.no_grad():
                latents_forget = encode_images(images_forget, vae)
                latents_retain = encode_images(images_retain, vae)
                text_embeds_forget = get_text_embeddings(prompts_forget, tokenizer, text_encoder)
                text_embeds_retain = get_text_embeddings(prompts_retain, tokenizer, text_encoder)
            
            # Compute loss with selected anchor strategy
            loss, loss_redirect, loss_preserve = compute_retrack_loss(
                unet, unet_pretrained, scheduler,
                latents_forget, latents_retain,
                text_embeds_forget, text_embeds_retain,
                anchor_indices=anchor_indices,
                text_embeds_anchor=text_embeds_anchor,
                epsilon=args.epsilon,
                lambda_stabilize=args.lambda_stabilize
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'redirect': f'{loss_redirect.item():.4f}',
                'preserve': f'{loss_preserve.item():.4f}'
            })
            
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_redirect': loss_redirect.item(),
                'train/loss_preserve': loss_preserve.item(),
                'train/step': global_step
            })
            
            # Generate sample images
            if global_step % args.sample_steps == 0 or global_step == 1:
                print(f"\nGenerating sample images at step {global_step}...")
                unet.eval()
                sample_dir = output_dir / "samples"
                images = generate_sample_images(
                    unet, vae, tokenizer, text_encoder,
                    args.forget_style, sample_dir, global_step
                )
                
                # Log to wandb with structured labels
                # Images are organized as: [forget_style x 3 objects, retain_style1 x 3 objects, ...]
                test_styles = [args.forget_style, "Impressionism", "Cubism", "Abstractionism"]
                test_objects = ["dog", "cat", "flower"]
                
                wandb_images = []
                idx = 0
                for style in test_styles:
                    for obj in test_objects:
                        caption = f"{style} - {obj}"
                        wandb_images.append(wandb.Image(images[idx], caption=caption))
                        idx += 1
                
                wandb.log({
                    f"samples/all_images": wandb_images,
                    f"samples/step": global_step
                })
                
                # Also log separately by style for easier comparison
                idx = 0
                for style in test_styles:
                    style_images = []
                    for obj in test_objects:
                        style_images.append(wandb.Image(images[idx], caption=obj))
                        idx += 1
                    wandb.log({
                        f"samples/{style.replace(' ', '_')}": style_images
                    })
                
                unet.train()
            
            # Save checkpoint
            if global_step in args.save_steps:
                print(f"\nSaving checkpoint at step {global_step}...")
                checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    'unet_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    pbar.close()
    
    # Save final metadata
    metadata = {
        'forget_style': args.forget_style,
        'num_steps': args.num_steps,
        'final_step': global_step,
        'training_date': datetime.now().isoformat(),
        'args': vars(args)
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    wandb.finish()
    print(f"\n✓ Training complete! Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
