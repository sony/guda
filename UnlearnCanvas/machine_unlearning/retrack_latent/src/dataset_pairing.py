"""
Dataset utilities for UnlearnCanvas with content-style pairing.
Supports indexing for same-content cross-style matching.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_index(data_root: str, output_path: str):
    """
    Create index mapping (object_name, image_idx) -> {style_name: filepath}.
    
    UnlearnCanvas structure:
        data_root/
            style_name/
                object_name/
                    image_idx.jpg
            Seed_Image/
                object_name/
                    image_idx.jpg
    """
    data_root = Path(data_root)
    index = {}
    
    # Get all style directories (excluding Seed_Image)
    style_dirs = [d for d in data_root.iterdir() 
                  if d.is_dir() and d.name != 'Seed_Image']
    
    print(f"Found {len(style_dirs)} style directories")
    
    for style_dir in style_dirs:
        style_name = style_dir.name
        
        # Iterate through object directories
        for object_dir in style_dir.iterdir():
            if not object_dir.is_dir():
                continue
                
            object_name = object_dir.name
            
            # Iterate through images
            for img_path in object_dir.glob('*.jpg'):
                image_idx = img_path.stem
                
                key = (object_name, image_idx)
                if key not in index:
                    index[key] = {}
                
                index[key][style_name] = str(img_path)
    
    # Convert tuple keys to string keys for JSON serialization
    json_index = {
        f"{obj}___{idx}": styles
        for (obj, idx), styles in index.items()
    }
    
    # Save index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_index, f, indent=2)
    
    print(f"Created index with {len(json_index)} content items")
    print(f"Saved to {output_path}")
    
    return json_index


def load_index(index_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Load index and convert string keys back to tuples."""
    with open(index_path, 'r') as f:
        json_index = json.load(f)
    
    index = {}
    for key_str, styles in json_index.items():
        obj, idx = key_str.split('___', 1)
        index[(obj, idx)] = styles
    
    return index


class UnlearnCanvasDataset(Dataset):
    """
    Dataset for UnlearnCanvas images with optional style filtering.
    """
    def __init__(
        self,
        data_root: str,
        styles: List[str],
        transform: Optional[transforms.Compose] = None,
        caption_mode: str = "content_only",
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory of UnlearnCanvas dataset
            styles: List of style names to include
            transform: Image transforms
            caption_mode: How to generate captions ('content_only', 'style_only', 'both')
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_root = Path(data_root)
        self.styles = styles
        self.transform = transform or self._default_transform()
        self.caption_mode = caption_mode
        
        # Collect all image paths
        self.samples = []
        for style in styles:
            style_dir = self.data_root / style
            if not style_dir.exists():
                print(f"Warning: Style directory not found: {style_dir}")
                continue
            
            for object_dir in style_dir.iterdir():
                if not object_dir.is_dir():
                    continue
                    
                object_name = object_dir.name
                
                for img_path in object_dir.glob('*.jpg'):
                    self.samples.append({
                        'path': str(img_path),
                        'style': style,
                        'object': object_name,
                        'image_idx': img_path.stem
                    })
                    
                    if max_samples and len(self.samples) >= max_samples:
                        break
                
                if max_samples and len(self.samples) >= max_samples:
                    break
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        print(f"Loaded {len(self.samples)} samples from {len(styles)} styles")
    
    def _default_transform(self):
        """Default image transformation for SD 1.5 (512x512)."""
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def _make_caption(self, sample: Dict) -> str:
        """Generate caption based on caption_mode."""
        if self.caption_mode == "content_only":
            # Only describe the object/content
            return f"a photo of {sample['object']}"
        elif self.caption_mode == "style_only":
            # Only mention style
            return f"in {sample['style']} style"
        elif self.caption_mode == "both":
            # Combine both
            return f"a photo of {sample['object']} in {sample['style']} style"
        else:
            raise ValueError(f"Unknown caption_mode: {self.caption_mode}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)
        
        # Generate caption
        caption = self._make_caption(sample)
        
        return {
            'pixel_values': image,
            'caption': caption,
            'style': sample['style'],
            'object': sample['object'],
            'image_idx': sample['image_idx']
        }


class PairedUnlearnDataset(Dataset):
    """
    Dataset for unlearning with paired samples:
    - Forget samples: from target style
    - Redirect samples: same content, different style
    """
    def __init__(
        self,
        data_root: str,
        index_path: str,
        forget_style: str,
        redirect_styles: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        caption_mode: str = "content_only",
        same_noise: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: Root directory of UnlearnCanvas dataset
            index_path: Path to content-style index JSON
            forget_style: Style to unlearn
            redirect_styles: Styles to redirect to (if None, use all except forget_style)
            transform: Image transforms
            caption_mode: How to generate captions
            same_noise: Whether to use same noise for paired samples
            max_samples: Maximum number of samples
        """
        self.data_root = Path(data_root)
        self.forget_style = forget_style
        self.transform = transform or self._default_transform()
        self.caption_mode = caption_mode
        self.same_noise = same_noise
        
        # Load index
        self.index = load_index(index_path)
        
        # Determine redirect styles
        all_styles = set()
        for styles_dict in self.index.values():
            all_styles.update(styles_dict.keys())
        
        if redirect_styles is None:
            redirect_styles = list(all_styles - {forget_style})
        
        self.redirect_styles = redirect_styles
        
        # Build paired samples
        self.pairs = []
        for (obj, idx), styles_dict in self.index.items():
            # Check if forget style exists
            if forget_style not in styles_dict:
                continue
            
            # Check if at least one redirect style exists
            available_redirects = [s for s in redirect_styles if s in styles_dict]
            if not available_redirects:
                continue
            
            # Create pairs (for now, use first available redirect)
            # TODO: Could extend to use multiple redirects or kNN selection
            redirect_style = available_redirects[0]
            
            self.pairs.append({
                'forget_path': styles_dict[forget_style],
                'redirect_path': styles_dict[redirect_style],
                'object': obj,
                'image_idx': idx,
                'forget_style': forget_style,
                'redirect_style': redirect_style
            })
            
            if max_samples and len(self.pairs) >= max_samples:
                break
        
        print(f"Created {len(self.pairs)} paired samples")
        print(f"Forget style: {forget_style}")
        print(f"Redirect styles: {redirect_styles}")
    
    def _default_transform(self):
        """Default image transformation for SD 1.5 (512x512)."""
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def _make_caption(self, obj: str, style: str) -> str:
        """Generate caption based on caption_mode."""
        if self.caption_mode == "content_only":
            return f"a photo of {obj}"
        elif self.caption_mode == "style_only":
            return f"in {style} style"
        elif self.caption_mode == "both":
            return f"a photo of {obj} in {style} style"
        else:
            raise ValueError(f"Unknown caption_mode: {self.caption_mode}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        forget_img = Image.open(pair['forget_path']).convert('RGB')
        redirect_img = Image.open(pair['redirect_path']).convert('RGB')
        
        forget_img = self.transform(forget_img)
        redirect_img = self.transform(redirect_img)
        
        # Generate captions
        forget_caption = self._make_caption(pair['object'], pair['forget_style'])
        redirect_caption = self._make_caption(pair['object'], pair['redirect_style'])
        
        result = {
            'forget_pixel_values': forget_img,
            'redirect_pixel_values': redirect_img,
            'forget_caption': forget_caption,
            'redirect_caption': redirect_caption,
            'object': pair['object'],
            'image_idx': pair['image_idx']
        }
        
        # For same noise training, return the same seed
        if self.same_noise:
            result['noise_seed'] = idx
        
        return result


class PairedUnlearnDatasetWithPrompts(Dataset):
    """
    Dataset for ReTrack-style unlearning with pre-generated prompts.
    Provides (forget_image, retain_image) pairs with same content but different styles.
    Uses Qwen-generated prompts from JSONL files.
    """
    def __init__(
        self,
        data_root: str,
        prompt_file: str,
        forget_style: str,
        k_retain: int = 1,
        retain_styles: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing UnlearnCanvas images
            prompt_file: Path to JSONL file with prompts (train_prompts.jsonl)
            forget_style: Style to forget (e.g., "Impressionism")
            k_retain: Number of retain images per forget image (default: 1)
            retain_styles: List of styles to use for retain images (None = all except forget)
            transform: Image transforms
            max_samples: Maximum number of samples (for debugging)
            seed: Random seed for reproducibility
        """
        import random
        from collections import defaultdict
        
        self.data_root = Path(data_root)
        self.forget_style = forget_style
        self.k_retain = k_retain
        self.transform = transform or self._default_transform()
        self.rng = random.Random(seed)
        
        # Always exclude Seed_Images
        EXCLUDED_DIRS = {'Seed_Images'}
        
        # Build index: {(object, idx): {style: (path, prompt)}}
        self.content_index = defaultdict(dict)
        self.forget_samples = []
        all_styles = set()
        
        # Load prompts from JSONL
        with open(prompt_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                
                # Extract style, object, idx from rel_path
                rel_path = data['rel_path']
                parts = rel_path.split('/')
                
                if len(parts) < 4:
                    continue
                
                # Parse: UnlearnCanvas/Style/Object/idx.jpg
                style = parts[1]
                obj_name = parts[2]
                img_name = parts[3]
                idx = img_name.split('.')[0]
                
                # Skip excluded directories
                if style in EXCLUDED_DIRS:
                    continue
                
                all_styles.add(style)
                
                # Construct full image path
                rel_path_from_root = '/'.join(parts[1:])
                img_path = self.data_root / rel_path_from_root
                
                if not img_path.exists():
                    continue
                
                # Build index
                content_key = (obj_name, idx)
                self.content_index[content_key][style] = {
                    'path': str(img_path),
                    'prompt': data['prompt'],
                    'image_id': data['image_id']
                }
                
                # Collect forget samples
                if style == forget_style:
                    self.forget_samples.append({
                        'content_key': content_key,
                        'path': str(img_path),
                        'prompt': data['prompt'],
                        'image_id': data['image_id']
                    })
        
        # Determine retain styles
        if retain_styles is None:
            self.retain_styles = list(all_styles - {forget_style} - EXCLUDED_DIRS)
        else:
            self.retain_styles = [s for s in retain_styles 
                                 if s != forget_style and s not in EXCLUDED_DIRS]
        
        if not self.retain_styles:
            raise ValueError(f"No retain styles available (forget_style={forget_style})")
        
        # Filter forget samples to only those with available retain pairs
        self.valid_samples = []
        for sample in self.forget_samples:
            content_key = sample['content_key']
            available_styles = [s for s in self.retain_styles 
                              if s in self.content_index[content_key]]
            
            if available_styles:
                sample['available_retain_styles'] = available_styles
                self.valid_samples.append(sample)
        
        if not self.valid_samples:
            raise ValueError(f"No valid paired samples for forget_style={forget_style}")
        
        # Limit samples if specified
        if max_samples is not None:
            self.valid_samples = self.valid_samples[:max_samples]
        
        print(f"PairedUnlearnDatasetWithPrompts initialized:")
        print(f"  Forget style: {forget_style}")
        print(f"  Retain styles: {len(self.retain_styles)} styles")
        print(f"  Valid paired samples: {len(self.valid_samples)}")
        print(f"  K retain: {k_retain}")
    
    def _default_transform(self):
        """Default image transform (512x512 for SD1.5)."""
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary with:
                - forget_image: Tensor (3, H, W)
                - retain_image: Tensor (3, H, W)
                - prompt: Text prompt (same for both)
        """
        sample = self.valid_samples[idx]
        content_key = sample['content_key']
        
        # Load forget image
        forget_img = Image.open(sample['path']).convert('RGB')
        forget_tensor = self.transform(forget_img)
        
        # Select retain style randomly
        available_styles = sample['available_retain_styles']
        retain_style = self.rng.choice(available_styles)
        retain_data = self.content_index[content_key][retain_style]
        
        # Load retain image
        retain_img = Image.open(retain_data['path']).convert('RGB')
        retain_tensor = self.transform(retain_img)
        
        return {
            'forget_image': forget_tensor,
            'retain_image': retain_tensor,
            'prompt': sample['prompt']  # Use same prompt for both
        }


def main():
    """CLI for creating index."""
    parser = argparse.ArgumentParser(description='Create UnlearnCanvas content-style index')
    parser.add_argument('--make-index', action='store_true', help='Create new index')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of UnlearnCanvas')
    parser.add_argument('--out', type=str, default='uc_index.json', help='Output path for index')
    
    args = parser.parse_args()
    
    if args.make_index:
        make_index(args.data_root, args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
