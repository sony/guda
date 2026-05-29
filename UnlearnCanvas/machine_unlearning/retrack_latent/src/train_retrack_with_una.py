#!/usr/bin/env python3
"""
ReTrack Unlearning with Integrated UNA Computation (FFSD Version)

This script combines ReTrack unlearning training with UNA score computation at specified steps.
Unlike the original train_retrack_ffsd.py, this does NOT save model checkpoints to disk,
only saves UNA attribution scores at each evaluation step.

Key differences from train_retrack_ffsd.py:
1. Loads all-class ELBO cache at startup
2. Evaluates UNA scores at specified steps (default: every 1000 steps)
3. Does NOT save model checkpoints (saves storage)
4. Only saves CSV files with UNA scores

For parameter sweeps with limited storage.
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
import csv
import numpy as np
from PIL import Image

# Add parent directory to path for imports
# train_retrack_with_una.py is in: machine_unlearning/retrack_latent/src/
# Need to go up 3 levels to reach workspace root: src/ -> retrack_latent/ -> machine_unlearning/ -> workspace_root/
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader

# Import ELBO utilities from evaluation/
from evaluation.elbo_utils import compute_delta_elbo_with_cache


# Patch CLIPTextModel for compatibility
_ORIG_CLIPTEXT_INIT = CLIPTextModel.__init__

def _patched_cliptext_init(self, config, *model_args, **model_kwargs):
    model_kwargs.pop("offload_state_dict", None)
    return _ORIG_CLIPTEXT_INIT(self, config, *model_args, **model_kwargs)

CLIPTextModel.__init__ = _patched_cliptext_init


def remove_style_from_prompt(prompt: str) -> str:
    """
    Remove style information from FFSD prompt.
    
    FFSD prompts: "{Subject Template}, artistic style featuring {descriptors}"
    Returns: "{Subject Template}"
    """
    separator = ", artistic style featuring "
    if separator in prompt:
        return prompt.split(separator)[0].strip()
    
    # Fallback for UC prompts
    import re
    pattern = r'\s+in\s+[^\s]+(?:\s+[^\s]+)*\s+style'
    if re.search(pattern, prompt, flags=re.IGNORECASE):
        prompt_no_style = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
        return prompt_no_style.strip()
    
    return prompt


def extract_descriptors_from_prompt(prompt: str) -> list:
    """Extract descriptor list from FFSD prompt."""
    separator = ", artistic style featuring "
    if separator in prompt:
        descriptor_part = prompt.split(separator)[1].strip()
        descriptors = [d.strip() for d in descriptor_part.split(',')]
        return descriptors
    return []


def extract_object_from_prompt(prompt: str) -> str:
    """Extract object/subject from FFSD prompt."""
    import re
    subject_template = remove_style_from_prompt(prompt)
    patterns = [
        r'A depiction of (.+)',
        r'a photo of (.+)',
        r'an image of (.+)',
        r'(.+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, subject_template, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return subject_template


@torch.no_grad()
def encode_descriptors_with_clip(
    descriptors: list,
    clip_processor,
    clip_model,
    device: str = "cuda"
) -> torch.Tensor:
    """Encode descriptors with CLIP and return averaged, normalized embedding."""
    if len(descriptors) == 0:
        return torch.zeros(768, device=device)
    
    text_inputs = clip_processor(
        text=descriptors,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    avg_embedding = text_features.mean(dim=0)
    avg_embedding = avg_embedding / avg_embedding.norm()
    return avg_embedding


def compute_style_distribution(
    forget_embedding: torch.Tensor,
    style_prototypes: torch.Tensor,
    style_names: list,
    forget_style_name: str,
    beta: float = 2.0,
    eta_uniform: float = 0.3,
    sampling_mode: str = "weighted",
    device: str = "cuda"
) -> torch.Tensor:
    """Compute style selection distribution π_s(s | c_f)."""
    forget_embedding = forget_embedding.to(device)
    style_prototypes = style_prototypes.to(device)
    forget_idx = style_names.index(forget_style_name)
    pi_uniform = torch.ones(len(style_names), device=device)
    pi_uniform[forget_idx] = 0.0
    pi_uniform = pi_uniform / pi_uniform.sum()

    if sampling_mode == "uniform":
        return pi_uniform

    logits = beta * torch.matmul(forget_embedding, style_prototypes.t())
    logits[forget_idx] = -float('inf')
    pi_softmax = torch.softmax(logits, dim=0)
    pi_mixed = (1 - eta_uniform) * pi_softmax + eta_uniform * pi_uniform
    return pi_mixed


def synthesize_retain_prompt(
    object_name: str,
    style_descriptors: list,
    num_anchor_descriptors: int = 3
) -> str:
    """Synthesize retain prompt using object and sampled style descriptors."""
    subject_template = f"A depiction of {object_name}"
    selected_descriptors = random.sample(
        style_descriptors,
        k=min(num_anchor_descriptors, len(style_descriptors))
    )
    descriptor_str = ", ".join(selected_descriptors)
    return f"{subject_template}, artistic style featuring {descriptor_str}"


class UnlearnDataset(Dataset):
    """Dataset for ReTrack unlearning with forget and retain sets."""
    
    def __init__(self, prompts_file: Path, forget_style: str, data_root: Path):
        self.data_root = data_root
        self.forget_style = forget_style
        
        self.forget_samples = []
        self.retain_samples = []
        self.retain_by_object_id = {}
        
        with open(prompts_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                rel_path = data['rel_path']
                prompt = data['prompt']
                
                parts = rel_path.split('/')
                if len(parts) >= 4:
                    style = parts[1]
                    object_name = parts[2]
                    image_id = parts[3].replace('.jpg', '')
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
                        key = (object_name, image_id)
                        if key not in self.retain_by_object_id:
                            self.retain_by_object_id[key] = []
                        self.retain_by_object_id[key].append(len(self.retain_samples) - 1)
        
        print(f"Dataset loaded:")
        print(f"  Forget ({forget_style}): {len(self.forget_samples)}")
        print(f"  Retain (others): {len(self.retain_samples)}")
    
    def __len__(self):
        return max(len(self.forget_samples), len(self.retain_samples))
    
    def __getitem__(self, idx):
        if idx % 2 == 0:
            sample_idx = (idx // 2) % len(self.forget_samples)
            sample = self.forget_samples[sample_idx]
            is_forget = True
        else:
            sample_idx = (idx // 2) % len(self.retain_samples)
            sample = self.retain_samples[sample_idx]
            is_forget = False
        
        image = Image.open(sample['image_path']).convert('RGB').resize((512, 512))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = 2.0 * image - 1.0
        
        return {
            'image': image,
            'prompt': sample['prompt'],
            'is_forget': is_forget,
            'style': sample['style'],
            'object': sample['object'],
            'image_id': sample['image_id']
        }


def collate_fn(batch):
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
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    text_encoder = text_encoder.to(device, dtype=torch.float16).eval()
    vae = vae.to(device, dtype=torch.float16).eval()
    unet = unet.to(device, dtype=torch.float32)
    
    return tokenizer, text_encoder, vae, unet, scheduler


def load_allclass_checkpoint(unet: UNet2DConditionModel, checkpoint_path: Path):
    if checkpoint_path.is_dir():
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if "unet_state_dict" in checkpoint:
            state_dict = checkpoint["unet_state_dict"]
        elif "unet" in checkpoint:
            state_dict = checkpoint["unet"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError(f"No UNet weights in checkpoint. Keys: {checkpoint.keys()}")
        
        unet.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded all-class checkpoint from {checkpoint_path}")


def create_pretrained_unet_copy(unet: UNet2DConditionModel, device: str = "cuda"):
    import copy
    unet_pretrained = copy.deepcopy(unet)
    unet_pretrained.eval()
    unet_pretrained.requires_grad_(False)
    unet_pretrained = unet_pretrained.to(device, dtype=torch.float32)
    print("✓ Created frozen pre-trained UNet copy")
    return unet_pretrained


@torch.no_grad()
def encode_images(images, vae):
    images = images.half()
    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    return latents.float()


@torch.no_grad()
def get_text_embeddings(prompts, tokenizer, text_encoder):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]
    return embeddings.float()


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
    """Compute ReTrack ε-redirect loss."""
    batch_size_forget = latents_forget.shape[0]
    batch_size_retain = latents_retain.shape[0]
    
    timesteps_forget = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (batch_size_forget,), device=latents_forget.device
    ).long()
    timesteps_retain = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (batch_size_retain,), device=latents_retain.device
    ).long()
    
    noise_forget = torch.randn_like(latents_forget)
    noise_retain = torch.randn_like(latents_retain)
    
    noisy_latents_forget = scheduler.add_noise(latents_forget, noise_forget, timesteps_forget)
    noisy_latents_retain = scheduler.add_noise(latents_retain, noise_retain, timesteps_retain)
    
    noise_pred_forget = unet(noisy_latents_forget, timesteps_forget, encoder_hidden_states=text_embeds_forget).sample
    
    with torch.no_grad():
        if anchor_indices is not None:
            text_embeds_anchor_selected = text_embeds_retain[anchor_indices]
            noise_pred_anchor = unet_pretrained(noisy_latents_forget, timesteps_forget, encoder_hidden_states=text_embeds_anchor_selected).sample
        else:
            noise_pred_anchor = unet_pretrained(noisy_latents_forget, timesteps_forget, encoder_hidden_states=text_embeds_anchor).sample
    
    loss_redirect = F.mse_loss(noise_pred_forget, noise_pred_anchor)
    
    noise_pred_retain = unet(noisy_latents_retain, timesteps_retain, encoder_hidden_states=text_embeds_retain).sample
    with torch.no_grad():
        noise_pred_retain_pretrained = unet_pretrained(noisy_latents_retain, timesteps_retain, encoder_hidden_states=text_embeds_retain).sample
    loss_preserve = F.mse_loss(noise_pred_retain, noise_pred_retain_pretrained)
    
    loss = loss_redirect + lambda_stabilize * loss_preserve
    
    return loss, loss_redirect, loss_preserve


@torch.no_grad()
def compute_una_scores(
    unet_unlearned,
    vae,
    tokenizer,
    text_encoder,
    scheduler,
    elbo_cache_data,
    eval_images_dir: Path,
    eval_prompts_dict,
    forget_style: str,
    batch_size: int = 8,
    use_amp: bool = True,
    amp_dtype = torch.bfloat16,
    device: str = "cuda"
):
    """
    Compute UNA scores using cached all-class ELBO.
    
    Returns:
        List of dicts with UNA scores for each image
    """
    unet_unlearned.eval()
    unet_unlearned = unet_unlearned.half()
    
    # Load cache data
    allclass_elbo = elbo_cache_data['elbo_values'].to(device)
    image_metadata = elbo_cache_data['image_metadata']
    noise_params = elbo_cache_data['noise_params']
    timesteps = elbo_cache_data['timesteps'].to(device)
    
    total_images = len(image_metadata)
    
    # Collect image paths
    all_image_paths = []
    for meta in image_metadata:
        img_path = eval_images_dir / meta['rel_path']
        all_image_paths.append(img_path)
    
    # Encode all images
    all_latents = []
    for i in range(0, total_images, batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        images = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images.append(img)
        
        images = torch.stack(images).to(device)
        images = 2.0 * images - 1.0
        images = images.half()
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        all_latents.append(latents.float())
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # Generate prompt embeddings
    all_prompt_embeds = []
    for i in range(0, total_images, batch_size):
        batch_metadata = image_metadata[i:i + batch_size]
        batch_prompts = []
        
        for meta in batch_metadata:
            style = meta['generated_style']
            obj = meta['object_name']
            prompt = eval_prompts_dict.get((style, obj))
            if not prompt:
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
    
    # Regenerate noise with same params as cache
    torch.manual_seed(noise_params['seed'])
    noise_dtype = amp_dtype if use_amp else torch.float32
    pre_noise = torch.randn(
        noise_params['num_noise_samples'],
        len(timesteps),
        total_images,
        4, 64, 64,
        device=device,
        dtype=noise_dtype
    )
    
    # Compute ΔELBO
    all_una_scores = []
    for batch_idx in range(0, total_images, batch_size):
        batch_end = min(batch_idx + batch_size, total_images)
        batch_latents = all_latents[batch_idx:batch_end]
        batch_prompt_embeds = all_prompt_embeds[batch_idx:batch_end]
        batch_noise = pre_noise[:, :, batch_idx:batch_end]
        batch_allclass_elbo = allclass_elbo[batch_idx:batch_end]
        
        delta_elbo = compute_delta_elbo_with_cache(
            latent=batch_latents,
            prompt_embeds=batch_prompt_embeds,
            unet_base=unet_unlearned,
            scheduler=scheduler,
            timesteps=timesteps,
            pre_noise=batch_noise,
            elbo_modified_cache=batch_allclass_elbo,
            num_samples=noise_params['num_noise_samples'],
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )
        all_una_scores.append(delta_elbo.cpu())
    
    all_una_scores = torch.cat(all_una_scores, dim=0)
    
    # Format results
    results = []
    for img_idx, (meta, score) in enumerate(zip(image_metadata, all_una_scores)):
        results.append({
            'generated_style': meta['generated_style'],
            'object_name': meta['object_name'],
            'attribution_style': forget_style,
            'una_score': score.item(),
            'image_path': meta['rel_path']
        })
    
    unet_unlearned.train()
    unet_unlearned = unet_unlearned.float()
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="ReTrack with integrated UNA computation")
    parser.add_argument("--allclass_checkpoint", type=str, required=True)
    parser.add_argument("--forget_style", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--elbo_cache_file", type=str, required=True, help="Path to precomputed all-class ELBO cache")
    parser.add_argument("--eval_images_dir", type=str, required=True, help="Directory with evaluation images")
    parser.add_argument("--eval_prompts_file", type=str, required=True, help="Evaluation prompts JSONL")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda_stabilize", type=float, default=2.0)
    parser.add_argument("--una_eval_steps", type=int, nargs='+', default=[1000, 2000, 3000, 4000, 5000])
    parser.add_argument("--sample_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="unlearn-canvas-retrack")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--anchor_strategy", type=str, default="style_removed", choices=["matched_retain", "style_removed", "weighted_style_select"])
    parser.add_argument("--beta", type=float, default=2.0, help="Temperature for weighted_style_select")
    parser.add_argument("--eta_uniform", type=float, default=0.3, help="Uniform mixing for weighted_style_select")
    parser.add_argument("--style_sampling_mode", type=str, default="weighted", choices=["weighted", "uniform"])
    parser.add_argument("--num_anchor_descriptors", type=int, default=3)
    parser.add_argument("--clip_prototypes_file", type=str, default="cache/clip_style_prototypes_very_relaxed.pt")
    parser.add_argument("--descriptors_file", type=str, default="data/ffsd_very_relaxed_descriptors.json")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for UNA evaluation")
    parser.add_argument("--param_hash", type=str, default=None, help="Hash ID for parameter configuration")
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    prompts_file = Path(args.prompts_file)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ELBO cache
    print(f"Loading all-class ELBO cache from {args.elbo_cache_file}...")
    elbo_cache_data = torch.load(args.elbo_cache_file, map_location="cpu")
    print(f"✓ Cache loaded: {len(elbo_cache_data['elbo_values'])} images")
    
    # Load evaluation prompts
    eval_prompts = {}
    with open(args.eval_prompts_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            style, obj, prompt = data['style'], data['object'], data['prompt']
            eval_prompts[(style, obj)] = prompt
            eval_prompts[(style.replace(' ', '_'), obj)] = prompt
    
    # Initialize wandb
    param_str = args.param_hash or f"lr{args.learning_rate:.0e}_lam{args.lambda_stabilize}_eps{args.epsilon}"
    wandb_name = args.wandb_name or f"retrack_{args.forget_style}_{param_str}"
    wandb.init(project=args.wandb_project, name=wandb_name, config=vars(args))
    
    # Load components
    print("Loading SD1.5 components...")
    tokenizer, text_encoder, vae, unet, scheduler = load_base_model(device="cuda")
    
    print("Loading all-class checkpoint...")
    load_allclass_checkpoint(unet, Path(args.allclass_checkpoint))
    unet_pretrained = create_pretrained_unet_copy(unet, device="cuda")
    
    # Load CLIP model and prototypes for weighted_style_select strategy
    clip_processor = None
    clip_model = None
    style_prototypes = None
    style_names = None
    style_descriptors_dict = None
    
    if args.anchor_strategy == "weighted_style_select":
        print("\nLoading CLIP model and style prototypes...")
        from transformers import CLIPModel, CLIPProcessor
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = clip_model.to("cuda")
        clip_model.eval()
        print("✓ CLIP model loaded")
        
        prototypes_file = Path(args.clip_prototypes_file)
        if not prototypes_file.exists():
            raise FileNotFoundError(
                f"CLIP prototypes file not found: {prototypes_file}\n"
                "Please create it with: python evaluation/precompute_clip_prototypes.py "
                "--descriptors_file data/ffsd_very_relaxed_descriptors.json "
                "--output_path cache/clip_style_prototypes_very_relaxed.pt"
            )
        
        prototype_data = torch.load(prototypes_file, map_location="cpu", weights_only=False)
        style_prototypes = prototype_data['prototypes']
        style_names = prototype_data['style_names']
        print(f"✓ Loaded style prototypes: {style_prototypes.shape}")
        
        descriptors_file = Path(args.descriptors_file)
        if not descriptors_file.exists():
            raise FileNotFoundError(f"Descriptors file not found: {descriptors_file}")
        
        with open(descriptors_file, 'r') as f:
            style_descriptors_dict = json.load(f)
        print(f"✓ Loaded descriptors for {len(style_descriptors_dict)} styles")
        print()
    
    # Setup training
    dataset = UnlearnDataset(prompts_file, args.forget_style, data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    unet.train()
    
    global_step = 0
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while global_step < args.num_steps:
        for batch in dataloader:
            if global_step >= args.num_steps:
                break
            
            images = batch['images'].to("cuda")
            prompts = batch['prompts']
            is_forget = batch['is_forget']
            objects = batch['objects']
            image_ids = batch['image_ids']
            
            forget_mask = is_forget
            retain_mask = ~is_forget
            
            if forget_mask.sum() == 0 or retain_mask.sum() == 0:
                continue
            
            images_forget = images[forget_mask]
            images_retain = images[retain_mask]
            prompts_forget = [p for p, f in zip(prompts, is_forget) if f]
            prompts_retain = [p for p, f in zip(prompts, is_forget) if not f]
            objects_forget = [o for o, f in zip(objects, is_forget) if f]
            objects_retain = [o for o, f in zip(objects, is_forget) if not f]
            ids_forget = [i for i, f in zip(image_ids, is_forget) if f]
            ids_retain = [i for i, f in zip(image_ids, is_forget) if not f]
            
            if args.anchor_strategy == "matched_retain":
                anchor_indices = []
                for obj_f, id_f in zip(objects_forget, ids_forget):
                    matches = [i for i, (obj_r, id_r) in enumerate(zip(objects_retain, ids_retain))
                              if obj_r == obj_f and id_r == id_f]
                    anchor_idx = random.choice(matches) if matches else random.randint(0, len(objects_retain) - 1)
                    anchor_indices.append(anchor_idx)
                text_embeds_anchor = None
            elif args.anchor_strategy == "style_removed":
                prompts_forget_no_style = [remove_style_from_prompt(p) for p in prompts_forget]
                anchor_indices = None
                with torch.no_grad():
                    text_embeds_anchor = get_text_embeddings(prompts_forget_no_style, tokenizer, text_encoder)
            else:  # weighted_style_select
                anchor_indices = None
                synthesized_prompts = []
                
                for prompt_f in prompts_forget:
                    descriptors_f = extract_descriptors_from_prompt(prompt_f)
                    object_f = extract_object_from_prompt(prompt_f)
                    
                    with torch.no_grad():
                        forget_embedding = encode_descriptors_with_clip(
                            descriptors_f, clip_processor, clip_model, device="cuda"
                        )
                    
                    style_dist = compute_style_distribution(
                        forget_embedding, style_prototypes, style_names, args.forget_style,
                        beta=args.beta,
                        eta_uniform=args.eta_uniform,
                        sampling_mode=args.style_sampling_mode,
                        device="cuda"
                    )
                    
                    sampled_style_idx = torch.multinomial(style_dist, num_samples=1).item()
                    sampled_style_name = style_names[sampled_style_idx]
                    sampled_descriptors = style_descriptors_dict[sampled_style_name]
                    
                    synthesized_prompt = synthesize_retain_prompt(
                        object_f,
                        sampled_descriptors,
                        num_anchor_descriptors=args.num_anchor_descriptors
                    )
                    synthesized_prompts.append(synthesized_prompt)
                
                with torch.no_grad():
                    text_embeds_anchor = get_text_embeddings(synthesized_prompts, tokenizer, text_encoder)
            
            with torch.no_grad():
                latents_forget = encode_images(images_forget, vae)
                latents_retain = encode_images(images_retain, vae)
                text_embeds_forget = get_text_embeddings(prompts_forget, tokenizer, text_encoder)
                text_embeds_retain = get_text_embeddings(prompts_retain, tokenizer, text_encoder)
            
            loss, loss_redirect, loss_preserve = compute_retrack_loss(
                unet, unet_pretrained, scheduler,
                latents_forget, latents_retain,
                text_embeds_forget, text_embeds_retain,
                anchor_indices=anchor_indices,
                text_embeds_anchor=text_embeds_anchor,
                epsilon=args.epsilon,
                lambda_stabilize=args.lambda_stabilize
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
            
            # Evaluate UNA scores at specified steps
            if global_step in args.una_eval_steps:
                print(f"\n{'='*80}")
                print(f"Computing UNA scores at step {global_step}...")
                print(f"{'='*80}")
                
                una_results = compute_una_scores(
                    unet,
                    vae,
                    tokenizer,
                    text_encoder,
                    scheduler,
                    elbo_cache_data,
                    Path(args.eval_images_dir),
                    eval_prompts,
                    args.forget_style,
                    batch_size=args.eval_batch_size,
                    use_amp=True,
                    amp_dtype=torch.bfloat16,
                    device="cuda"
                )
                
                # Save UNA scores to CSV
                una_csv_path = output_dir / f"una_scores_step{global_step}.csv"
                with open(una_csv_path, 'w', newline='') as csvfile:
                    fieldnames = ['generated_style', 'object_name', 'attribution_style', 'una_score', 'image_path']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(una_results)
                
                print(f"✓ UNA scores saved to {una_csv_path}")
                print(f"  Total scores: {len(una_results)}")
                print(f"{'='*80}\n")
                
                wandb.log({
                    f"una/step_{global_step}_mean": np.mean([r['una_score'] for r in una_results]),
                    f"una/step_{global_step}_std": np.std([r['una_score'] for r in una_results]),
                })
    
    pbar.close()
    
    # Save final metadata
    metadata = {
        'forget_style': args.forget_style,
        'num_steps': args.num_steps,
        'final_step': global_step,
        'training_date': datetime.now().isoformat(),
        'args': vars(args),
        'una_eval_steps': args.una_eval_steps,
        'param_hash': args.param_hash,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    wandb.finish()
    print(f"\n✓ Training complete! UNA scores saved to: {output_dir}")


if __name__ == "__main__":
    main()
