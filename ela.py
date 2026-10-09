"""
Error Level Analysis (ELA) Module.

ELA reveals JPEG compression inconsistencies by re-saving an image at a known
quality and computing the pixel-wise difference.  Forged regions that were
pasted from a different JPEG source show up as brighter areas in the ELA image
because their error levels differ from the surrounding authentic content.

Two components:
    ELATransform  — PIL image → ELA PIL image (used in data pipeline)
    ELAEncoder    — EfficientNet-B0 backbone operating on ELA images
"""

import io

import numpy as np
from PIL import Image, ImageChops


class ELATransform:
    """
    Compute Error Level Analysis image from a PIL image.

    Pipeline:
        1. Re-save original at JPEG quality ``quality`` into memory buffer
        2. Compute |original − recompressed| per pixel
        3. Scale × ``scale`` and clip to [0, 255]
        4. Return as 3-channel RGB PIL image

    The resulting image makes forgery boundaries *glow* — different
    compression histories produce different error levels.
    """

    def __init__(self, quality: int = 90, scale: float = 15.0):
        self.quality = quality
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        # Ensure RGB
        img = img.convert("RGB")

        # Re-save at fixed quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # Pixel-wise absolute difference
        diff = ImageChops.difference(img, recompressed)

        # Scale up to make differences visible
        diff_np = np.array(diff, dtype=np.float32)
        diff_np = np.clip(diff_np * self.scale, 0, 255).astype(np.uint8)

        return Image.fromarray(diff_np)

    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality}, scale={self.scale})"
