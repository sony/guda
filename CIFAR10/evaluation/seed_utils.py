#!/usr/bin/env python3
"""
Seed utilities for deterministic ELBO computation.

Ensures reproducibility by generating deterministic seeds from image paths.
This allows the same noise to be used across different runs when evaluating
the same images, regardless of execution order or batch configuration.
"""
import hashlib
import torch
from typing import List, Union
from pathlib import Path


def get_deterministic_seed_for_image(image_path: str, base_seed: int = 42) -> int:
    """
    Generate deterministic seed from image path.
    
    Uses only the filename (not full path) to ensure portability across
    different directory structures.
    
    Args:
        image_path: Path to image (e.g., 'results/.../image_0001.png')
        base_seed: Base seed for variation (default: 42)
    
    Returns:
        Deterministic 32-bit integer seed
    
    Example:
        >>> seed1 = get_deterministic_seed_for_image("image_0001.png", 42)
        >>> seed2 = get_deterministic_seed_for_image("/other/path/image_0001.png", 42)
        >>> assert seed1 == seed2  # Same filename → same seed
    """
    # Use only filename to ensure portability
    filename = Path(image_path).name
    hash_input = f"{filename}_{base_seed}".encode()
    hash_digest = hashlib.sha256(hash_input).hexdigest()
    # Convert to 32-bit integer for torch.manual_seed compatibility
    return int(hash_digest, 16) % (2**32)


def set_seed_for_batch(
    image_paths: List[Union[str, Path]], 
    base_seed: int = 42
) -> None:
    """
    Set deterministic seed for a batch based on first image path.
    
    This ensures that:
    1. Same batch always gets same seed (reproducibility)
    2. Different batches get different seeds (independence)
    3. Seed is portable across different systems
    
    Args:
        image_paths: List of image paths in batch
        base_seed: Base seed for variation
    
    Example:
        >>> batch = [Path("image_0001.png"), Path("image_0002.png")]
        >>> set_seed_for_batch(batch, 42)
        >>> # Now torch.randn() will generate deterministic noise
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    
    # Use first image in batch to determine seed
    seed = get_deterministic_seed_for_image(str(image_paths[0]), base_seed)
    
    # Set seeds for both CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Self-test
    print("Testing seed_utils...")
    
    # Test 1: Same filename → same seed
    seed1 = get_deterministic_seed_for_image("image_0001.png", 42)
    seed2 = get_deterministic_seed_for_image("/different/path/image_0001.png", 42)
    assert seed1 == seed2, "Same filename should produce same seed"
    print(f"✓ Test 1 passed: seed={seed1}")
    
    # Test 2: Different filename → different seed
    seed3 = get_deterministic_seed_for_image("image_0002.png", 42)
    assert seed1 != seed3, "Different filename should produce different seed"
    print(f"✓ Test 2 passed: seed1={seed1}, seed3={seed3}")
    
    # Test 3: Different base_seed → different seed
    seed4 = get_deterministic_seed_for_image("image_0001.png", 123)
    assert seed1 != seed4, "Different base_seed should produce different seed"
    print(f"✓ Test 3 passed: seed1={seed1}, seed4={seed4}")
    
    # Test 4: Batch seeding
    batch = [Path("image_0001.png"), Path("image_0002.png")]
    set_seed_for_batch(batch, 42)
    val1 = torch.randn(1).item()
    
    set_seed_for_batch(batch, 42)
    val2 = torch.randn(1).item()
    assert abs(val1 - val2) < 1e-6, "Same batch should produce same random values"
    print(f"✓ Test 4 passed: reproducible random value={val1:.6f}")
    
    print("\n✓ All tests passed!")
