#!/usr/bin/env python3
"""
Precompute CLIP Style Prototypes for weighted_style_select Anchor Strategy

This script:
1. Loads style descriptors from ffsd_very_relaxed_descriptors.json
2. Encodes each descriptor with CLIP text encoder
3. Averages descriptor embeddings per style (individual average strategy)
4. L2-normalizes to create style prototypes
5. Saves 60 × 768 tensor to cache/

Usage:
    python evaluation/precompute_clip_prototypes.py \
        --descriptors_file data/ffsd_very_relaxed_descriptors.json \
        --output_file cache/clip_style_prototypes_very_relaxed.pt \
        --clip_model openai/clip-vit-large-patch14
"""
import argparse
import json
import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def load_descriptors(descriptors_file: Path) -> dict:
    """Load style descriptors from JSON file."""
    with open(descriptors_file, 'r') as f:
        descriptors = json.load(f)
    return descriptors


@torch.no_grad()
def encode_descriptor_list(
    descriptors: list,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode a list of descriptors and return individual embeddings.
    
    Args:
        descriptors: List of descriptor strings
        processor: CLIP processor
        model: CLIP model
        device: Device to run on
    
    Returns:
        Tensor of shape (num_descriptors, 768) with L2-normalized embeddings
    """
    # Tokenize and encode
    text_inputs = processor(
        text=descriptors,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # Get CLIP text features
    text_features = model.get_text_features(**text_inputs)
    
    # L2 normalize each descriptor embedding
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu()


def compute_style_prototype(
    descriptor_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute style prototype from descriptor embeddings.
    
    Args:
        descriptor_embeddings: Tensor of shape (num_descriptors, 768)
    
    Returns:
        L2-normalized prototype vector of shape (768,)
    """
    # Average descriptor embeddings
    prototype = descriptor_embeddings.mean(dim=0)
    
    # L2 normalize
    prototype = prototype / prototype.norm()
    
    return prototype


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP style prototypes")
    parser.add_argument(
        "--descriptors_file",
        type=str,
        default="data/ffsd_very_relaxed_descriptors.json",
        help="JSON file with style descriptors"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="cache/clip_style_prototypes_very_relaxed.pt",
        help="Output file for style prototypes"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    args = parser.parse_args()
    
    # Setup paths
    descriptors_file = Path(args.descriptors_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CLIP Style Prototype Precomputation")
    print("=" * 80)
    print(f"Descriptors file: {descriptors_file}")
    print(f"Output file: {output_file}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Device: {args.device}")
    print()
    
    # Load descriptors
    print("Loading descriptors...")
    descriptors_dict = load_descriptors(descriptors_file)
    style_names = sorted(descriptors_dict.keys())
    num_styles = len(style_names)
    print(f"✓ Loaded {num_styles} styles")
    print(f"  Example: {style_names[0]} -> {descriptors_dict[style_names[0]]}")
    print()
    
    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}...")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model)
    model = model.to(args.device)
    model.eval()
    print("✓ CLIP model loaded")
    print()
    
    # Compute prototypes for each style
    print("Computing style prototypes...")
    prototypes = []
    
    for style_name in tqdm(style_names, desc="Processing styles"):
        descriptors = descriptors_dict[style_name]
        
        # Encode individual descriptors
        descriptor_embeddings = encode_descriptor_list(
            descriptors,
            processor,
            model,
            device=args.device
        )
        
        # Compute prototype (average + L2 normalize)
        prototype = compute_style_prototype(descriptor_embeddings)
        prototypes.append(prototype)
    
    # Stack into single tensor: (num_styles, 768)
    prototypes_tensor = torch.stack(prototypes, dim=0)
    
    print(f"✓ Computed {num_styles} prototypes")
    print(f"  Shape: {prototypes_tensor.shape}")
    print(f"  Dtype: {prototypes_tensor.dtype}")
    print()
    
    # Save to file
    print(f"Saving prototypes to {output_file}...")
    save_data = {
        'prototypes': prototypes_tensor,
        'style_names': style_names,
        'clip_model': args.clip_model,
        'num_styles': num_styles,
        'embedding_dim': prototypes_tensor.shape[1]
    }
    torch.save(save_data, output_file)
    print("✓ Saved successfully")
    print()
    
    # Verification
    print("Verification:")
    print(f"  All prototypes normalized: {torch.allclose(prototypes_tensor.norm(dim=1), torch.ones(num_styles))}")
    print(f"  Mean prototype norm: {prototypes_tensor.norm(dim=1).mean():.6f}")
    print(f"  Min/Max prototype norm: {prototypes_tensor.norm(dim=1).min():.6f} / {prototypes_tensor.norm(dim=1).max():.6f}")
    
    # Sample pairwise similarities
    similarities = torch.mm(prototypes_tensor, prototypes_tensor.t())
    # Set diagonal to -inf to ignore self-similarity
    similarities.fill_diagonal_(-float('inf'))
    print(f"  Max pairwise similarity: {similarities.max():.4f}")
    print(f"  Min pairwise similarity: {similarities.min():.4f}")
    print(f"  Mean pairwise similarity: {similarities[similarities != -float('inf')].mean():.4f}")
    print()
    
    print("=" * 80)
    print("✓ Prototype precomputation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
