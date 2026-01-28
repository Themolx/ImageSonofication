"""Core sonification algorithms."""

from img2sound.core.scanline import scanline_image, scanline_video
from img2sound.core.spectral import spectral_image, spectral_video
from img2sound.core.additive import additive_image, additive_video
from img2sound.core.audio import normalize_audio, write_wav, resample_audio, auto_scale_audio

__all__ = [
    "scanline_image",
    "scanline_video",
    "spectral_image",
    "spectral_video",
    "additive_image",
    "additive_video",
    "normalize_audio",
    "write_wav",
    "resample_audio",
    "auto_scale_audio",
]
