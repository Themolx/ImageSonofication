"""img2sound - Convert images and video to audio."""

__version__ = "0.1.0"

from img2sound.core.scanline import scanline_image, scanline_video
from img2sound.core.spectral import spectral_image, spectral_video
from img2sound.core.audio import normalize_audio, write_wav

__all__ = [
    "scanline_image",
    "scanline_video",
    "spectral_image",
    "spectral_video",
    "normalize_audio",
    "write_wav",
]
