"""Tests for ForensicDataset."""
import os
import shutil
from pathlib import Path

from PIL import Image

from data.dataset import ForensicDataset
from data.transforms import get_val_transforms


ARTIFACT_DIR = Path("tests") / "_dataset_artifacts"


def _create_test_dataset(tmp_dir, n_per_class=3):
    """Create a minimal test dataset directory structure."""
    for split in ["train", "val"]:
        for label in ["Authentic", "CopyMove", "Splicing"]:
            class_dir = os.path.join(tmp_dir, split, label)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(n_per_class):
                img = Image.new("RGB", (100, 100), color="blue")
                img.save(os.path.join(class_dir, f"{label}_{i:03d}.jpg"))


def _prepare_dataset_dir(name: str) -> Path:
    target = ARTIFACT_DIR / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def test_dataset_loading():
    dataset_dir = _prepare_dataset_dir("loading")
    try:
        _create_test_dataset(str(dataset_dir), n_per_class=3)
        ds = ForensicDataset(
            root_dir=str(dataset_dir),
            split="train",
            transform=get_val_transforms(224, use_clip=False),
        )
        assert len(ds) == 9  # 3 classes × 3 images
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def test_dataset_getitem():
    dataset_dir = _prepare_dataset_dir("getitem")
    try:
        _create_test_dataset(str(dataset_dir), n_per_class=2)
        ds = ForensicDataset(
            root_dir=str(dataset_dir),
            split="train",
            transform=get_val_transforms(224, use_clip=False),
        )
        image, x_clip, label = ds[0]
        assert image.shape == (3, 224, 224)
        assert x_clip is None
        assert label in [0, 1, 2]
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def test_class_counts():
    dataset_dir = _prepare_dataset_dir("counts")
    try:
        _create_test_dataset(str(dataset_dir), n_per_class=5)
        ds = ForensicDataset(root_dir=str(dataset_dir), split="train")
        counts = ds.get_class_counts()
        assert counts["Authentic"] == 5
        assert counts["CopyMove"] == 5
        assert counts["Splicing"] == 5
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)
