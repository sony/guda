"""
Data manifest management for CIFAR-10 training groups.

Provides structured access to training data with class or group labels. These
helpers are used by LOGO and unlearning trainers when a custom group manifest is
provided.
"""

import hashlib
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class TrainManifest:
    """Manifest for training dataset."""
    train_ids: List[int]  # Training example IDs
    group_ids: List[int]  # Group (class) labels
    dataset_name: str      # 'cifar10' or 'cifar100'
    split: str = 'train'
    
    def __len__(self):
        return len(self.train_ids)
    
    def save(self, path: Path):
        """Save manifest to pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(asdict(self), f)
        print(f"Saved train manifest to {path} ({len(self)} examples)")
    
    @classmethod
    def load(cls, path: Path):
        """Load manifest from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)
    
    def get_group_indices(self, group_id: int) -> np.ndarray:
        """Get training indices belonging to a specific group."""
        return np.array([i for i, gid in enumerate(self.group_ids) if gid == group_id])
    
    def get_group_sizes(self, num_groups: int = 10) -> Dict[int, int]:
        """Get size of each group."""
        return {gid: sum(1 for g in self.group_ids if g == gid) for gid in range(num_groups)}


@dataclass
class QueryManifest:
    """Manifest for query (generated) samples."""
    query_ids: List[str]     # Query identifiers (e.g., "image_0000")
    image_paths: List[str]   # Paths to generated images (optional, can be None)
    seeds: List[int]         # Random seeds used for generation
    prompts: Optional[List[str]] = None  # For conditional generation (SD1.5)
    
    def __len__(self):
        return len(self.query_ids)
    
    def save(self, path: Path):
        """Save manifest to pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(asdict(self), f)
        print(f"Saved query manifest to {path} ({len(self)} queries)")
    
    @classmethod
    def load(cls, path: Path):
        """Load manifest from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)
    
    def subset(self, indices: List[int]) -> 'QueryManifest':
        """Create a subset of queries."""
        return QueryManifest(
            query_ids=[self.query_ids[i] for i in indices],
            image_paths=[self.image_paths[i] for i in indices] if self.image_paths else None,
            seeds=[self.seeds[i] for i in indices],
            prompts=[self.prompts[i] for i in indices] if self.prompts else None
        )


def create_cifar10_train_manifest(
    data_dir: Path = Path('./data'),
    output_path: Optional[Path] = None
) -> TrainManifest:
    """
    Create training manifest for CIFAR-10.
    
    Args:
        data_dir: Directory containing CIFAR-10 data
        output_path: Optional path to save manifest
    
    Returns:
        TrainManifest with 50,000 training examples
    """
    from evaluation.attribution_utils import load_group_ids_from_dataset
    
    group_ids = load_group_ids_from_dataset('cifar10', 'train', data_dir)
    train_ids = list(range(len(group_ids)))
    
    manifest = TrainManifest(
        train_ids=train_ids,
        group_ids=group_ids.tolist(),
        dataset_name='cifar10',
        split='train'
    )
    
    if output_path:
        manifest.save(output_path)
    
    return manifest


def create_query_manifest_from_generated(
    image_dir: Path,
    num_queries: int,
    seeds: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    image_pattern: str = "sample_{:04d}.png"
) -> QueryManifest:
    """
    Create query manifest from generated images directory.
    
    Args:
        image_dir: Directory containing generated images
        num_queries: Number of query images
        seeds: Random seeds (if None, use sequential IDs)
        output_path: Optional path to save manifest
        image_pattern: Filename pattern (supports .format())
    
    Returns:
        QueryManifest
    """
    query_ids = [f"image_{i:04d}" for i in range(num_queries)]
    image_paths = [str(image_dir / image_pattern.format(i)) for i in range(num_queries)]
    
    if seeds is None:
        seeds = list(range(num_queries))
    
    manifest = QueryManifest(
        query_ids=query_ids,
        image_paths=image_paths,
        seeds=seeds
    )
    
    if output_path:
        manifest.save(output_path)
    
    return manifest


def create_query_manifest_subset(
    full_manifest_path: Path,
    subset_size: int = 10,
    seed: int = 42,
    output_path: Optional[Path] = None
) -> QueryManifest:
    """
    Create a small subset of queries for validation.
    
    Args:
        full_manifest_path: Path to full query manifest
        subset_size: Number of queries in subset (default: 10 for validation)
        seed: Random seed for selection
        output_path: Optional path to save subset manifest
    
    Returns:
        QueryManifest (subset)
    """
    full_manifest = QueryManifest.load(full_manifest_path)
    
    np.random.seed(seed)
    indices = np.random.choice(len(full_manifest), size=subset_size, replace=False)
    indices = sorted(indices.tolist())
    
    subset_manifest = full_manifest.subset(indices)
    
    if output_path:
        subset_manifest.save(output_path)
    
    return subset_manifest


def load_train_manifest_checked(
    manifest_path: Path,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    expected_len: Optional[int] = None,
) -> TrainManifest:
    """
    Load a train manifest and validate basic structural expectations.
    """
    manifest = TrainManifest.load(manifest_path)

    if dataset_name is not None and manifest.dataset_name != dataset_name:
        raise ValueError(
            f"Manifest dataset mismatch: expected {dataset_name}, got {manifest.dataset_name}"
        )
    if split is not None and manifest.split != split:
        raise ValueError(
            f"Manifest split mismatch: expected {split}, got {manifest.split}"
        )
    if expected_len is not None and len(manifest) != expected_len:
        raise ValueError(
            f"Manifest length mismatch: expected {expected_len}, got {len(manifest)}"
        )

    return manifest


def stable_manifest_hash(manifest: TrainManifest, length: int = 12) -> str:
    """
    Build a short stable hash for cache keys and output tagging.
    """
    payload = hashlib.sha256()
    payload.update(manifest.dataset_name.encode("utf-8"))
    payload.update(b"\0")
    payload.update(manifest.split.encode("utf-8"))
    payload.update(b"\0")
    payload.update(np.asarray(manifest.train_ids, dtype=np.int64).tobytes())
    payload.update(np.asarray(manifest.group_ids, dtype=np.int64).tobytes())
    return payload.hexdigest()[:length]


def select_first_train_ids_per_group(
    *,
    train_ids: List[int],
    group_ids: List[int],
    num_groups: int,
    num_samples_per_group: int,
) -> Dict[int, List[int]]:
    """
    Return the first `num_samples_per_group` train ids encountered for each group.
    """
    selected: Dict[int, List[int]] = {group_id: [] for group_id in range(num_groups)}
    for train_id, group_id in zip(train_ids, group_ids):
        group_id = int(group_id)
        if group_id not in selected:
            continue
        if len(selected[group_id]) < num_samples_per_group:
            selected[group_id].append(int(train_id))
        if all(len(ids) >= num_samples_per_group for ids in selected.values()):
            break
    missing_groups = [
        group_id
        for group_id, ids in selected.items()
        if len(ids) < num_samples_per_group
    ]
    if missing_groups:
        raise ValueError(
            "Could not collect enough train ids for groups "
            f"{missing_groups}; requested {num_samples_per_group} per group"
        )
    return selected


def _random_derangement(num_items: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random derangement of range(num_items).
    """
    if num_items < 2:
        raise ValueError("Derangement requires at least 2 items")
    while True:
        perm = rng.permutation(num_items)
        if np.all(perm != np.arange(num_items)):
            return perm


def create_balanced_noisy_cifar10_train_manifest(
    *,
    data_dir: Path = Path("./data"),
    noise_per_class: int = 250,
    seed: int = 42,
    block_size: int = 25,
    output_path: Optional[Path] = None,
    assignment_csv_path: Optional[Path] = None,
    transition_json_path: Optional[Path] = None,
    transition_csv_path: Optional[Path] = None,
) -> Tuple[TrainManifest, np.ndarray]:
    """
    Create a CIFAR-10 train manifest with balanced 5%-noise group assignment.

    Procedure:
    1. Sample `noise_per_class` train examples from each original class.
    2. Split them into equal-size blocks (`block_size`, default 25).
    3. For each block index, sample a 10-way derangement and move the whole block.

    Guarantees:
    - Each source class moves exactly `noise_per_class` examples.
    - Each destination class receives exactly `noise_per_class` examples.
    - No moved example is reassigned to its original class.
    """
    if noise_per_class <= 0:
        raise ValueError("noise_per_class must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if noise_per_class % block_size != 0:
        raise ValueError("noise_per_class must be divisible by block_size")

    base_manifest = create_cifar10_train_manifest(data_dir=data_dir)
    train_ids = np.asarray(base_manifest.train_ids, dtype=np.int64)
    original_group_ids = np.asarray(base_manifest.group_ids, dtype=np.int64)
    noisy_group_ids = original_group_ids.copy()

    num_groups = 10
    blocks_per_class = noise_per_class // block_size
    rng = np.random.default_rng(seed)

    source_class_to_ids: Dict[int, np.ndarray] = {}
    for class_id in range(num_groups):
        class_train_ids = train_ids[original_group_ids == class_id]
        if len(class_train_ids) < noise_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_train_ids)} examples; "
                f"cannot sample {noise_per_class}"
            )
        sampled_ids = rng.choice(class_train_ids, size=noise_per_class, replace=False)
        rng.shuffle(sampled_ids)
        source_class_to_ids[class_id] = sampled_ids

    transition_matrix = np.zeros((num_groups, num_groups), dtype=np.int64)
    moved_ids = []

    for block_idx in range(blocks_per_class):
        block_perm = _random_derangement(num_groups, rng)
        for source_class in range(num_groups):
            start = block_idx * block_size
            end = start + block_size
            block_ids = source_class_to_ids[source_class][start:end]
            destination_class = int(block_perm[source_class])

            noisy_group_ids[block_ids] = destination_class
            transition_matrix[source_class, destination_class] += block_size
            moved_ids.extend(int(idx) for idx in block_ids.tolist())

    if len(set(moved_ids)) != num_groups * noise_per_class:
        raise ValueError("Moved train ids are not unique")
    if not np.all(transition_matrix.sum(axis=1) == noise_per_class):
        raise ValueError("Transition matrix row sums are invalid")
    if not np.all(transition_matrix.sum(axis=0) == noise_per_class):
        raise ValueError("Transition matrix column sums are invalid")
    if not np.all(np.diag(transition_matrix) == 0):
        raise ValueError("Transition matrix diagonal must be zero")
    group_sizes = np.bincount(noisy_group_ids, minlength=num_groups)
    if not np.all(group_sizes == 5000):
        raise ValueError(f"Noisy group sizes are invalid: {group_sizes.tolist()}")

    manifest = TrainManifest(
        train_ids=train_ids.tolist(),
        group_ids=noisy_group_ids.tolist(),
        dataset_name="cifar10",
        split="train",
    )

    if output_path is not None:
        manifest.save(output_path)

    if assignment_csv_path is not None:
        assignment_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(assignment_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["train_id", "original_label", "noisy_group_id", "is_moved"])
            for train_id, original_label, noisy_label in zip(
                train_ids.tolist(),
                original_group_ids.tolist(),
                noisy_group_ids.tolist(),
            ):
                writer.writerow(
                    [
                        int(train_id),
                        int(original_label),
                        int(noisy_label),
                        int(original_label != noisy_label),
                    ]
                )

    transition_payload = {
        "seed": int(seed),
        "noise_per_class": int(noise_per_class),
        "block_size": int(block_size),
        "blocks_per_class": int(blocks_per_class),
        "transition_matrix": transition_matrix.tolist(),
        "group_sizes": group_sizes.tolist(),
        "manifest_hash": stable_manifest_hash(manifest),
    }
    if transition_json_path is not None:
        transition_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transition_json_path, "w", encoding="utf-8") as handle:
            json.dump(transition_payload, handle, indent=2)

    if transition_csv_path is not None:
        transition_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(transition_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_class", *[f"dest_{i}" for i in range(num_groups)]])
            for source_class in range(num_groups):
                writer.writerow([source_class, *transition_matrix[source_class].tolist()])

    return manifest, transition_matrix


class ManifestDataset(Dataset):
    """
    PyTorch Dataset wrapper for manifests.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        manifest: TrainManifest,
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            base_dataset: Base PyTorch dataset (e.g., CIFAR10)
            manifest: TrainManifest
            indices: Optional subset of indices to use
        """
        self.base_dataset = base_dataset
        self.manifest = manifest
        self.indices = indices if indices is not None else list(range(len(manifest)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label, train_id, group_id)
        """
        train_idx = self.indices[idx]
        image, label = self.base_dataset[train_idx]
        train_id = self.manifest.train_ids[train_idx]
        group_id = self.manifest.group_ids[train_idx]
        return image, label, train_id, group_id


def split_indices_for_parallel(
    num_examples: int,
    num_workers: int,
    worker_id: int
) -> List[int]:
    """
    Split indices for parallel processing.
    
    Args:
        num_examples: Total number of examples
        num_workers: Number of parallel workers
        worker_id: Current worker ID (0-indexed)
    
    Returns:
        List of indices for this worker
    """
    all_indices = np.arange(num_examples)
    indices_per_worker = np.array_split(all_indices, num_workers)
    return indices_per_worker[worker_id].tolist()


def get_cifar10_dataloader(
    manifest: TrainManifest,
    batch_size: int = 128,
    num_workers: int = 4,
    indices: Optional[List[int]] = None
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader from manifest for CIFAR-10.
    
    Args:
        manifest: TrainManifest
        batch_size: Batch size
        num_workers: Number of data loading workers
        indices: Optional subset of indices
    
    Returns:
        DataLoader
    """
    import torchvision
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if manifest.dataset_name == 'cifar100':
        DatasetClass = torchvision.datasets.CIFAR100
    elif manifest.dataset_name == 'cifar10':
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f"Unsupported dataset in manifest: {manifest.dataset_name}")

    base_dataset = DatasetClass(
        root='./data',
        train=(manifest.split == 'train'),
        download=False,
        transform=transform
    )
    
    dataset = ManifestDataset(base_dataset, manifest, indices)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain order for parallel processing
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
