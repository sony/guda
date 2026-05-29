#!/usr/bin/env python3
"""
Precompute mean CLIP embeddings for all training images.

This script computes and saves the mean CLIP embedding for each style's training images.
This is a preprocessing step that enables fast CLIPA score computation.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.eval_utils import load_style_images, get_all_styles


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for training images")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/UnlearnCanvas",
        help="Root directory containing training images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/clip_embeddings",
        help="Directory to save precomputed embeddings"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--max_training_images",
        type=int,
        default=400,
        help="Maximum number of training images per style"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--styles",
        nargs='+',
        default=None,
        help="Specific styles to process (default: all)"
    )
    return parser.parse_args()


@torch.no_grad()
def encode_images_clip(
    image_paths: list,
    processor: CLIPProcessor,
    model: CLIPModel,
    batch_size: int = 8,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode multiple images with CLIP.
    
    Returns:
        Tensor of shape (N, D) where N is number of images, D is embedding dimension
    """
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load and process images
        images = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(images) == 0:
            continue
        
        # Encode with CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.get_image_features(**inputs)
        # Normalize embeddings
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        all_embeddings.append(embeddings.cpu())
    
    if len(all_embeddings) == 0:
        return torch.empty(0, model.config.vision_config.hidden_size)
    
    return torch.cat(all_embeddings, dim=0)


def main():
    args = parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get styles to process
    if args.styles:
        styles = args.styles
    else:
        styles = get_all_styles(data_root)
    
    print(f"Precomputing CLIP embeddings for {len(styles)} styles")
    print(f"CLIP model: {args.clip_model}")
    print(f"Max training images per style: {args.max_training_images}")
    print(f"Output directory: {output_dir}\n")
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()
    print(f"CLIP model loaded on {device}\n")
    
    # Process each style
    for style_idx, style in enumerate(tqdm(styles, desc="Processing styles")):
        try:
            # Load training images
            train_image_paths = load_style_images(
                data_root,
                style,
                max_images=args.max_training_images
            )
            
            if len(train_image_paths) == 0:
                print(f"Warning: No training images found for style: {style}")
                continue
            
            print(f"  [{style_idx+1}/{len(styles)}] {style}: {len(train_image_paths)} images")
            
            # Encode all images
            embeddings = encode_images_clip(
                train_image_paths,
                processor,
                model,
                batch_size=args.batch_size,
                device=device
            )
            
            if embeddings.shape[0] == 0:
                print(f"  Warning: No embeddings computed for {style}")
                continue
            
            # Compute mean embedding
            mean_embedding = embeddings.mean(dim=0)
            # Normalize the mean embedding
            mean_embedding = mean_embedding / mean_embedding.norm()
            
            # Also compute all individual embeddings for max aggregation if needed
            embeddings_normalized = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Save both mean and individual embeddings
            output_data = {
                'style': style,
                'num_images': len(train_image_paths),
                'mean_embedding': mean_embedding.numpy(),
                'all_embeddings': embeddings_normalized.numpy(),
                'embedding_dim': embeddings.shape[1]
            }
            
            output_path = output_dir / f"{style.replace(' ', '_')}.npz"
            np.savez_compressed(output_path, **output_data)
            
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {style}: {e}")
            continue
    
    print(f"\nPrecomputation complete!")
    print(f"Embeddings saved to: {output_dir}")
    print(f"Total styles processed: {len(styles)}")


if __name__ == "__main__":
    main()
