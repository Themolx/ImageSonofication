"""Input/output utilities for images and video."""

from img2sound.io.image import load_image, to_grayscale
from img2sound.io.video import extract_frames, get_video_info

__all__ = [
    "load_image",
    "to_grayscale",
    "extract_frames",
    "get_video_info",
]
