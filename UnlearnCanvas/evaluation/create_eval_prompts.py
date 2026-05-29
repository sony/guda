#!/usr/bin/env python3
"""
Create evaluation prompts that are distinct from training prompts.

This script generates evaluation prompts for LOGOA/UNA/Wang attribution methods
by extracting descriptors from training prompts and creating new prompts with:
1. Different template structure
2. Shuffled descriptor order
3. Different phrasing

This ensures evaluation prompts do not match training data.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_style_object_descriptors(
    train_prompts_file: Path
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract descriptors from training prompts.
    
    Returns:
        dict: {style: {object: [descriptor1, descriptor2, ...]}}
    """
    style_obj_descriptors = defaultdict(lambda: defaultdict(list))
    
    print(f"Extracting descriptors from {train_prompts_file}...")
    
    with open(train_prompts_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            style = data['style']
            obj = data['object']
            prompt = data['prompt']
            
            # Extract descriptors after "artistic style featuring"
            if 'artistic style featuring ' in prompt:
                desc_str = prompt.split('artistic style featuring ')[1].strip()
                # Remove trailing period if present
                desc_str = desc_str.rstrip('.')
                
                # Split by comma
                descriptors = [d.strip() for d in desc_str.split(',')]
                
                # Store only if not already present for this (style, object)
                if obj not in style_obj_descriptors[style]:
                    style_obj_descriptors[style][obj] = descriptors
    
    print(f"Extracted descriptors for {len(style_obj_descriptors)} styles")
    
    return style_obj_descriptors


def create_eval_prompts(
    style_obj_descriptors: Dict[str, Dict[str, List[str]]],
    target_styles: List[str],
    seed: int = 42
) -> List[dict]:
    """
    Create evaluation prompts distinct from training prompts.
    
    Strategy:
    1. Use different template: "An artwork of {obj} in {style} style, with {desc}"
    2. Shuffle descriptor order (deterministically based on style+object)
    3. Ensure no overlap with training prompts
    
    Args:
        style_obj_descriptors: Extracted descriptors from training data
        target_styles: List of styles to generate prompts for
        seed: Random seed for reproducibility
        
    Returns:
        List of eval prompt dicts
    """
    eval_prompts = []
    
    print(f"\nCreating evaluation prompts for {len(target_styles)} styles...")
    
    for style in target_styles:
        if style not in style_obj_descriptors:
            print(f"Warning: No descriptors found for style '{style}'")
            continue
        
        objects = sorted(style_obj_descriptors[style].keys())
        
        for obj in objects:
            descriptors = style_obj_descriptors[style][obj].copy()
            
            # Shuffle descriptors deterministically
            # Use hash of (style, object) as seed for reproducibility
            local_seed = hash(style + obj) % (2**32)
            random.seed(local_seed)
            random.shuffle(descriptors)
            
            # Create descriptor string
            desc_str = ", ".join(descriptors)
            
            # Use different template from training data (WITHOUT style name)
            # Training: "A depiction of {obj}, artistic style featuring {desc}"
            # Eval: "An artwork of {obj}, with {desc}"
            # Note: Style name is NOT included to match training conventions
            prompt = f"An artwork of {obj}, with {desc_str}"
            
            eval_prompts.append({
                "style": style,
                "object": obj,
                "prompt": prompt,
                "source": "eval",
                "descriptors": descriptors,
                "descriptor_order": "shuffled"
            })
    
    print(f"Created {len(eval_prompts)} evaluation prompts")
    
    return eval_prompts


def verify_no_overlap(
    train_prompts_file: Path,
    eval_prompts: List[dict]
) -> bool:
    """
    Verify that evaluation prompts do not overlap with training prompts.
    
    Returns:
        True if no overlap, False otherwise
    """
    print("\nVerifying no overlap with training data...")
    
    # Load all training prompts
    train_prompts_set = set()
    with open(train_prompts_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            train_prompts_set.add(data['prompt'])
    
    # Check eval prompts
    eval_prompts_set = {item['prompt'] for item in eval_prompts}
    
    overlap = train_prompts_set & eval_prompts_set
    
    if overlap:
        print(f"❌ ERROR: Found {len(overlap)} overlapping prompts!")
        print("Examples:")
        for prompt in list(overlap)[:3]:
            print(f"  - {prompt}")
        return False
    else:
        print(f"✓ No overlap found!")
        print(f"  Training prompts: {len(train_prompts_set)}")
        print(f"  Evaluation prompts: {len(eval_prompts_set)}")
        return True


def save_eval_prompts(eval_prompts: List[dict], output_file: Path):
    """Save evaluation prompts to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in eval_prompts:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Saved {len(eval_prompts)} prompts to {output_file}")


def print_sample_comparison(
    train_prompts_file: Path,
    eval_prompts: List[dict],
    num_samples: int = 3
):
    """Print sample training vs evaluation prompts for comparison."""
    print("\n" + "=" * 80)
    print("Sample Training vs Evaluation Prompts")
    print("=" * 80)
    
    # Load first few training prompts
    train_samples = []
    with open(train_prompts_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            train_samples.append(json.loads(line))
    
    for i in range(min(num_samples, len(eval_prompts))):
        train = train_samples[i]
        eval_item = eval_prompts[i]
        
        print(f"\n{i+1}. Style: {train['style']}, Object: {train['object']}")
        print(f"   Training: {train['prompt'][:100]}...")
        print(f"   Eval:     {eval_item['prompt'][:100]}...")
        
        # Check if descriptors are different
        if 'artistic style featuring ' in train['prompt']:
            train_desc = train['prompt'].split('artistic style featuring ')[1].strip()
            eval_desc = ', '.join(eval_item['descriptors'])
            if train_desc != eval_desc:
                print(f"   ✓ Descriptors shuffled")


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation prompts distinct from training data"
    )
    parser.add_argument(
        "--train_prompts_file",
        type=str,
        default="data/UnlearnCanvas/train_prompts_ffsd_very_relaxed.jsonl",
        help="Training prompts file (source for descriptors)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/UnlearnCanvas/eval_prompts_ffsd_very_relaxed.jsonl",
        help="Output evaluation prompts file"
    )
    parser.add_argument(
        "--styles",
        nargs='+',
        default=[
            "Abstractionism", "Artist Sketch", "Blossom Season", "Blue Blooming",
            "Bricks", "Byzantine", "Cartoon", "Cold Warm",
            "Color Fantasy", "Comic Etch", "Crayon", "Crypto Punks",
            "Cubism", "Dadaism", "Dapple", "Defoliation"
        ],
        help="Styles to generate prompts for (default: 16 A-D styles)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for descriptor shuffling"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify no overlap with training data"
    )
    
    args = parser.parse_args()
    
    train_prompts_file = Path(args.train_prompts_file)
    output_file = Path(args.output_file)
    
    print("=" * 80)
    print("Creating Evaluation Prompts")
    print("=" * 80)
    print(f"Training prompts: {train_prompts_file}")
    print(f"Output file: {output_file}")
    print(f"Target styles: {len(args.styles)}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Step 1: Extract descriptors from training data
    style_obj_descriptors = extract_style_object_descriptors(train_prompts_file)
    
    # Step 2: Create evaluation prompts
    eval_prompts = create_eval_prompts(
        style_obj_descriptors,
        args.styles,
        args.seed
    )
    
    # Step 3: Verify no overlap
    if args.verify:
        no_overlap = verify_no_overlap(train_prompts_file, eval_prompts)
        if not no_overlap:
            print("\n❌ Verification failed! Not saving prompts.")
            return 1
    
    # Step 4: Print sample comparison
    print_sample_comparison(train_prompts_file, eval_prompts)
    
    # Step 5: Save prompts
    save_eval_prompts(eval_prompts, output_file)
    
    print("\n" + "=" * 80)
    print("✓ Evaluation prompts created successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
