#!/usr/bin/env python3
"""
Create index file for UnlearnCanvas dataset.
Maps (object_name, image_idx) -> {style: filepath}
"""

import json
from pathlib import Path
import argparse


def make_index(data_root: str, output_path: str):
    """
    Create index mapping (object_name, image_idx) -> {style_name: filepath}.
    
    UnlearnCanvas structure:
        data_root/UnlearnCanvas/
            style_name/
                object_name/
                    image_idx.jpg
    """
    data_root = Path(data_root) / "UnlearnCanvas"
    index = {}
    
    # Get all style directories
    style_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
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
                
                key = f"{object_name}___{image_idx}"
                if key not in index:
                    index[key] = {}
                
                index[key][style_name] = str(img_path)
    
    # Save index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Created index with {len(index)} content items")
    print(f"Average styles per content: {sum(len(v) for v in index.values()) / len(index):.1f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create UnlearnCanvas index')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing UnlearnCanvas/')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for index JSON')
    
    args = parser.parse_args()
    make_index(args.data_root, args.output)
