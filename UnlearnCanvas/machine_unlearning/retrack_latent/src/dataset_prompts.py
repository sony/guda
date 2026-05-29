"""
Dataset for UnlearnCanvas using pre-generated prompts from JSONL files.
This replaces the simple caption generation with high-quality Qwen-generated prompts.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PromptDataset(Dataset):
    """
    Dataset that loads images and uses pre-generated prompts from JSONL.
    
    Expected JSONL format:
    {
        "image_id": "UnlearnCanvas/Style/Object/idx",
        "rel_path": "UnlearnCanvas/Style/Object/idx.jpg",
        "prompt": "content description. style elements."
    }
    """
    
    def __init__(
        self,
        data_root: str,
        prompt_file: str,
        styles: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory containing UnlearnCanvas images
            prompt_file: Path to JSONL file with prompts (train_prompts.jsonl)
            styles: If provided, only load images from these styles
            exclude_styles: If provided, exclude images from these styles (LOGO)
            transform: Image transforms
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.transform = transform or self._default_transform()

        # Always exclude Seed_Images (not used for training)
        EXCLUDED_DIRS = {'Seed_Images', 'Seed Images'}
        if exclude_styles is None:
            exclude_styles = list(EXCLUDED_DIRS)
        else:
            exclude_styles = list(set(exclude_styles) | EXCLUDED_DIRS)
        
        # Load prompts from JSONL
        self.samples = []
        with open(prompt_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                
                # Extract style from rel_path: UnlearnCanvas/Style/Object/idx.jpg
                rel_path = data['rel_path']
                parts = rel_path.split('/')
                if len(parts) >= 2:
                    style = parts[1]  # Style is the second part
                else:
                    continue
                
                # Filter by style if specified
                if styles is not None and style not in styles:
                    continue
                
                # Exclude styles if specified (LOGO)
                if exclude_styles is not None and style in exclude_styles:
                    continue
                
                # Construct full image path
                # rel_path includes "UnlearnCanvas/..." but data_root already points to UnlearnCanvas/
                # So we need to skip the first part
                if parts[0] == 'UnlearnCanvas':
                    rel_path_from_root = '/'.join(parts[1:])
                else:
                    rel_path_from_root = rel_path
                    
                img_path = self.data_root / rel_path_from_root
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue
                
                self.samples.append({
                    'path': str(img_path),
                    'prompt': data['prompt'],
                    'style': style,
                    'image_id': data['image_id']
                })
                
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        print(f"Loaded {len(self.samples)} samples with prompts from {prompt_file}")
        if styles:
            print(f"Filtered to styles: {styles}")
    
    def _default_transform(self):
        """Default image transformation for SD 1.5 (512x512)."""
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'pixel_values': image,
            'caption': sample['prompt'],
            'style': sample['style'],
            'image_id': sample['image_id']
        }


class PairedPromptDataset(Dataset):
    """
    Paired dataset for unlearning: matches forget samples with redirect samples
    having the same content but different style.
    
    Uses pre-generated index for efficient pairing.
    """
    
    def __init__(
        self,
        data_root: str,
        prompt_file: str,
        index_path: str,
        forget_style: str,
        redirect_styles: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        k_redirect: int = 1,
        same_noise: bool = True
    ):
        """
        Args:
            data_root: Root directory containing images
            prompt_file: Path to train_prompts.jsonl
            index_path: Path to index JSON (created by make_index)
            forget_style: Style to unlearn
            redirect_styles: Styles to redirect to (if None, use all except forget)
            transform: Image transforms
            k_redirect: Number of redirect samples per forget sample
            same_noise: Whether to use same noise for paired samples
        """
        self.data_root = Path(data_root)
        self.forget_style = forget_style
        self.k_redirect = k_redirect
        self.same_noise = same_noise
        self.transform = transform or self._default_transform()
        
        # Load prompts
        self.prompts = {}  # image_id -> prompt
        with open(prompt_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.prompts[data['image_id']] = data['prompt']
        
        # Load index
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        # Get all styles
        all_styles = set()
        for styles_dict in self.index.values():
            all_styles.update(styles_dict.keys())
        
        if redirect_styles is None:
            redirect_styles = [s for s in all_styles if s != forget_style]
        
        self.redirect_styles = redirect_styles
        
        # Build paired samples
        self.pairs = []
        for content_key, styles_dict in self.index.items():
            # Must have forget style
            if forget_style not in styles_dict:
                continue
            
            forget_path = styles_dict[forget_style]
            forget_id = self._path_to_id(forget_path)
            
            if forget_id not in self.prompts:
                continue
            
            # Find redirect candidates
            redirect_candidates = []
            for style in redirect_styles:
                if style in styles_dict:
                    redirect_path = styles_dict[style]
                    redirect_id = self._path_to_id(redirect_path)
                    if redirect_id in self.prompts:
                        redirect_candidates.append({
                            'path': redirect_path,
                            'id': redirect_id,
                            'style': style
                        })
            
            if not redirect_candidates:
                continue
            
            # Sample k_redirect candidates
            import random
            sampled = random.sample(
                redirect_candidates,
                min(k_redirect, len(redirect_candidates))
            )
            
            for redirect in sampled:
                self.pairs.append({
                    'forget_path': forget_path,
                    'forget_id': forget_id,
                    'redirect_path': redirect['path'],
                    'redirect_id': redirect['id'],
                    'redirect_style': redirect['style']
                })
        
        print(f"Created {len(self.pairs)} paired samples")
        print(f"Forget style: {forget_style}")
        print(f"Redirect styles: {redirect_styles}")
    
    def _path_to_id(self, path: str) -> str:
        """Convert file path to image_id."""
        # /path/to/UnlearnCanvas/Style/Object/idx.jpg -> UnlearnCanvas/Style/Object/idx
        parts = Path(path).parts
        # Find UnlearnCanvas in path
        try:
            uc_idx = parts.index('UnlearnCanvas')
            rel_parts = parts[uc_idx:]
            # Remove .jpg extension
            image_id = '/'.join(rel_parts)
            image_id = image_id.rsplit('.', 1)[0]
            return image_id
        except (ValueError, IndexError):
            return path
    
    def _default_transform(self):
        """Default image transformation for SD 1.5 (512x512)."""
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load forget image
        forget_img = Image.open(pair['forget_path']).convert('RGB')
        forget_img = self.transform(forget_img)
        
        # Load redirect image
        redirect_img = Image.open(pair['redirect_path']).convert('RGB')
        redirect_img = self.transform(redirect_img)
        
        return {
            'forget_pixel_values': forget_img,
            'forget_caption': self.prompts[pair['forget_id']],
            'redirect_pixel_values': redirect_img,
            'redirect_caption': self.prompts[pair['redirect_id']],
            'redirect_style': pair['redirect_style']
        }
