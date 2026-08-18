import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "UCMerced_LandUse" / "Images"


class ApplyTransformDataset(torch.utils.data.Dataset):
    """Apply a transform to a subset while preserving the source dataset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _validate_classes(actual_classes, expected_classes, dataset_root):
    if expected_classes is None:
        from config import CLASS_NAMES

        expected_classes = CLASS_NAMES
    if tuple(actual_classes) != tuple(expected_classes):
        raise ValueError(
            "Dataset classes do not match config.CLASS_NAMES. "
            f"Expected {list(expected_classes)}, found {list(actual_classes)} in {dataset_root}"
        )


def prepare_data_loaders(
    backbone_name,
    data_root=None,
    batch_size=64,
    num_workers=0,
    seed=42,
    pin_memory=None,
    expected_classes=None,
):
    """Build deterministic, stratified train/validation/test data loaders."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    dataset_root = Path(data_root or DEFAULT_DATASET_ROOT).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"UC Merced image directory does not exist: {dataset_root}. "
            "Pass --data-root or place the dataset in the documented default directory."
        )

    base_size = 224 if any(name in backbone_name.lower() for name in ("vit", "swin")) else 256
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(base_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(base_size + 32),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )

    full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)
    _validate_classes(full_dataset.classes, expected_classes, dataset_root)

    targets = np.asarray(full_dataset.targets)
    indices = np.arange(len(full_dataset))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        stratify=targets[train_val_idx],
        random_state=seed,
    )

    datasets_by_split = {
        "train": ApplyTransformDataset(Subset(full_dataset, train_idx), train_transform),
        "val": ApplyTransformDataset(Subset(full_dataset, val_idx), evaluation_transform),
        "test": ApplyTransformDataset(Subset(full_dataset, test_idx), evaluation_transform),
    }
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    return {
        "train": DataLoader(
            datasets_by_split["train"],
            shuffle=True,
            generator=_generator(seed),
            **common_options,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            shuffle=False,
            generator=_generator(seed + 1),
            **common_options,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            shuffle=False,
            generator=_generator(seed + 2),
            **common_options,
        ),
    }
