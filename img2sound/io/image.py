"""Image loading and preprocessing utilities."""

import numpy as np
from PIL import Image
from pathlib import Path

from img2sound.utils.helpers import IMAGE_EXTENSIONS, UnsupportedFormatError, ProcessingError


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as RGB numpy array.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array (H, W, 3) in RGB format

    Raises:
        UnsupportedFormatError: If file extension not supported
        ProcessingError: If image cannot be loaded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported image format: {ext}")

    try:
        with Image.open(path) as img:
            # Convert to RGB if necessary
            if img.mode == "RGBA":
                # Handle transparency by compositing over white
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise ProcessingError(f"Failed to load image: {e}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale using luminosity method.

    Args:
        image: RGB image array (H, W, 3) or already grayscale (H, W)

    Returns:
        Grayscale image as numpy array (H, W)
    """
    if image.ndim == 2:
        # Already grayscale
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    if image.ndim == 3 and image.shape[2] >= 3:
        # RGB to grayscale using luminosity formula
        # Y = 0.299*R + 0.587*G + 0.114*B
        return (
            0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        ).astype(image.dtype)

    raise ValueError(f"Unexpected image shape: {image.shape}")


def validate_image(image: np.ndarray) -> None:
    """
    Validate image array for processing.

    Args:
        image: Image array to validate

    Raises:
        ProcessingError: If image is invalid
    """
    if image is None:
        raise ProcessingError("Image is None")

    if image.size == 0:
        raise ProcessingError("Image is empty")

    if image.ndim < 2 or image.ndim > 3:
        raise ProcessingError(f"Invalid image dimensions: {image.ndim}")

    h, w = image.shape[:2]
    if h < 2 or w < 2:
        raise ProcessingError(f"Image too small: {w}x{h}")
