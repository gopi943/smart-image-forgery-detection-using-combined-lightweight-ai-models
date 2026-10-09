"""Tests for input validation module."""
import os
import tempfile

from PIL import Image

from data.validation import ImageValidator


def test_valid_jpeg():
    validator = ImageValidator()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="red")
        img.save(f.name)
        result = validator.validate(f.name)
        assert result.valid
        assert result.image is not None
    os.unlink(f.name)


def test_rgba_conversion():
    validator = ImageValidator()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(f.name)
        result = validator.validate(f.name)
        assert result.valid
        assert result.image.mode == "RGB"
    os.unlink(f.name)


def test_unsupported_format():
    validator = ImageValidator()
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
        img = Image.new("RGB", (100, 100))
        img.save(f.name)
        result = validator.validate(f.name)
        assert not result.valid
        assert any("Unsupported" in e for e in result.errors)
    os.unlink(f.name)


def test_nonexistent_file():
    validator = ImageValidator()
    result = validator.validate("/nonexistent/file.jpg")
    assert not result.valid


def test_oversized_image():
    validator = ImageValidator(max_resolution=200)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (500, 500))
        img.save(f.name)
        result = validator.validate(f.name)
        assert not result.valid
    os.unlink(f.name)
