"""Adversarial and malformed input tests for the validation layer."""

from __future__ import annotations

import os
import shutil
import struct
import uuid
from pathlib import Path

import pytest
from PIL import Image

from data.validation import ImageValidator


ARTIFACT_DIR = Path("tests") / "_adversarial_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _make_dir(name: str) -> Path:
    path = ARTIFACT_DIR / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_path(name: str, suffix: str) -> Path:
    directory = _make_dir(name)
    return directory / f"sample{suffix}"


@pytest.fixture
def validator():
    return ImageValidator(
        supported_formats=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        max_resolution=4096,
        max_memory_mb=500.0,
    )


def _cleanup(path: Path) -> None:
    shutil.rmtree(path.parent, ignore_errors=True)


def test_truncated_jpeg(validator):
    path = _make_path("truncated", ".jpg")
    Image.new("RGB", (100, 100), color="red").save(path)

    try:
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        result = validator.validate(str(path))
        assert isinstance(result.valid, bool)
    finally:
        _cleanup(path)


def test_zero_byte_file(validator):
    path = _make_path("zero_byte", ".jpg")
    path.write_bytes(b"")

    try:
        result = validator.validate(str(path))
        assert not result.valid
        assert result.errors
    finally:
        _cleanup(path)


def test_random_binary_as_jpeg(validator):
    path = _make_path("random_binary", ".jpg")
    path.write_bytes(os.urandom(1024))

    try:
        result = validator.validate(str(path))
        assert not result.valid
    finally:
        _cleanup(path)


def test_png_header_but_jpeg_extension(validator):
    path = _make_path("png_header", ".jpg")
    Image.new("RGB", (50, 50), color="green").save(path, format="PNG")

    try:
        result = validator.validate(str(path))
        assert isinstance(result.valid, bool)
    finally:
        _cleanup(path)


def test_extremely_wide_image(validator):
    path = _make_path("wide", ".png")
    Image.new("RGB", (50000, 1), color="white").save(path)

    try:
        result = validator.validate(str(path))
        assert not result.valid
    finally:
        _cleanup(path)


def test_extremely_tall_image(validator):
    path = _make_path("tall", ".png")
    Image.new("RGB", (1, 50000), color="white").save(path)

    try:
        result = validator.validate(str(path))
        assert not result.valid
    finally:
        _cleanup(path)


def test_1x1_pixel_image(validator):
    path = _make_path("one_pixel", ".png")
    Image.new("RGB", (1, 1), color="red").save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
        assert result.image is not None
    finally:
        _cleanup(path)


def test_square_at_max_resolution():
    validator = ImageValidator(max_resolution=100)
    path = _make_path("max_resolution", ".png")
    Image.new("RGB", (100, 100)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
    finally:
        _cleanup(path)


def test_one_pixel_over_max():
    validator = ImageValidator(max_resolution=100)
    path = _make_path("over_max_resolution", ".png")
    Image.new("RGB", (101, 100)).save(path)

    try:
        result = validator.validate(str(path))
        assert not result.valid
    finally:
        _cleanup(path)


def test_grayscale_image(validator):
    path = _make_path("grayscale", ".png")
    Image.new("L", (100, 100), color=128).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
        assert result.image.mode == "RGB"
    finally:
        _cleanup(path)


def test_rgba_image(validator):
    path = _make_path("rgba", ".png")
    Image.new("RGBA", (100, 100), color=(255, 0, 0, 128)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
        assert result.image.mode == "RGB"
    finally:
        _cleanup(path)


def test_palette_mode_image(validator):
    path = _make_path("palette", ".png")
    Image.new("P", (100, 100)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
        assert result.image.mode == "RGB"
    finally:
        _cleanup(path)


def test_16bit_image(validator):
    path = _make_path("i16", ".png")
    Image.new("I;16", (100, 100)).save(path)

    try:
        result = validator.validate(str(path))
        assert isinstance(result.valid, bool)
    finally:
        _cleanup(path)


def test_nonexistent_path(validator):
    result = validator.validate("/this/does/not/exist/image.jpg")
    assert not result.valid


def test_directory_path(validator):
    directory = _make_dir("directory_path")
    try:
        result = validator.validate(str(directory))
        assert not result.valid
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_empty_string_path(validator):
    result = validator.validate("")
    assert not result.valid


def test_unicode_filename(validator):
    directory = _make_dir("unicode_filename")
    path = directory / "nono_unicode_test.png"
    Image.new("RGB", (50, 50), color="blue").save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_memory_guard_triggers():
    validator = ImageValidator(max_resolution=5000, max_memory_mb=1.0)
    path = _make_path("memory_guard", ".png")
    Image.new("RGB", (4000, 4000)).save(path)

    try:
        result = validator.validate(str(path))
        assert not result.valid
        assert any(
            "memory" in error.lower() or "tensor" in error.lower()
            for error in result.errors
        )
    finally:
        _cleanup(path)


def test_memory_guard_allows_small():
    validator = ImageValidator(max_memory_mb=100.0)
    path = _make_path("memory_small", ".png")
    Image.new("RGB", (100, 100)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
    finally:
        _cleanup(path)


def test_batch_validation_mixed(validator):
    valid_png = _make_path("batch_valid_png", ".png")
    zero_byte = _make_path("batch_zero_byte", ".jpg")
    valid_jpg = _make_path("batch_valid_jpg", ".jpg")

    Image.new("RGB", (50, 50)).save(valid_png)
    zero_byte.write_bytes(b"")
    Image.new("RGB", (50, 50)).save(valid_jpg)

    paths = [
        str(valid_png),
        str(zero_byte),
        str(valid_jpg),
        "/does/not/exist.jpg",
    ]

    try:
        results = validator.validate_batch(paths)
        assert len(results) == 4
        assert results[0].valid is True
        assert results[1].valid is False
        assert results[2].valid is True
        assert results[3].valid is False
    finally:
        _cleanup(valid_png)
        _cleanup(zero_byte)
        _cleanup(valid_jpg)


def test_unsupported_extension(validator):
    path = _make_path("unsupported_ext", ".webp")
    Image.new("RGB", (50, 50)).save(path, format="WEBP")

    try:
        result = validator.validate(str(path))
        assert isinstance(result.valid, bool)
    finally:
        _cleanup(path)


def test_case_insensitive_extension(validator):
    path = _make_path("case_insensitive", ".JPG")
    Image.new("RGB", (50, 50)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
    finally:
        _cleanup(path)


def test_double_extension(validator):
    directory = _make_dir("double_extension")
    path = directory / "image.tar.jpg"
    Image.new("RGB", (50, 50)).save(path)

    try:
        result = validator.validate(str(path))
        assert result.valid
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_structured_garbage_header(validator):
    path = _make_path("structured_garbage", ".jpg")
    path.write_bytes(struct.pack(">I", 0xFFD8FFE0) + os.urandom(64))

    try:
        result = validator.validate(str(path))
        assert isinstance(result.valid, bool)
    finally:
        _cleanup(path)
