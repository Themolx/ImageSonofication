"""Scanline sonification - spatial mapping of brightness to amplitude."""

import numpy as np
from typing import Generator

from img2sound.io.image import to_grayscale
from img2sound.io.video import extract_frames, get_video_info
from img2sound.core.audio import (
    resample_audio,
    auto_scale_audio,
    apply_bass_boost,
    apply_lowpass,
    apply_highpass,
    apply_smooth,
)


def scanline_image(
    image: np.ndarray,
    direction: str = "horizontal",
    channel_mode: str = "mono",
    invert: bool = False,
) -> np.ndarray:
    """
    Convert image to audio using scanline sonification.

    Each column (or row) becomes a moment in time. Brightness values
    at each pixel become amplitude values.

    Args:
        image: RGB or grayscale image array (H, W, C) or (H, W)
        direction: 'horizontal' (L->R) or 'vertical' (T->B)
        channel_mode: 'mono', 'stereo', or 'rgb'
        invert: If True, dark pixels become loud

    Returns:
        Audio samples as np.ndarray, shape (N,) for mono, (N, 2) for stereo
    """
    img = image.astype(np.float32)

    if channel_mode == "mono":
        img = to_grayscale(img)
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if direction == "vertical":
        if img.ndim == 2:
            img = np.transpose(img, (1, 0))
        else:
            img = np.transpose(img, (1, 0, 2))

    if channel_mode == "mono":
        audio = img.T.flatten()
    elif channel_mode == "stereo":
        left = img[:, :, 0].T.flatten()
        right = img[:, :, 2].T.flatten()
        audio = np.stack([left, right], axis=1)
    elif channel_mode == "rgb":
        r = img[:, :, 0].T.flatten()
        g = img[:, :, 1].T.flatten()
        b = img[:, :, 2].T.flatten()
        audio = np.stack([r, g, b], axis=1)
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    audio = auto_scale_audio(audio)

    if invert:
        audio = -audio

    return audio.astype(np.float32)


def process_scanline_audio(
    audio: np.ndarray,
    sample_rate: int,
    bass_boost: float = 1.0,
    lowpass: int = None,
    highpass: int = None,
    smooth: int = 0,
) -> np.ndarray:
    """
    Apply audio processing to scanline output.

    Args:
        audio: Raw scanline audio
        sample_rate: Sample rate in Hz
        bass_boost: Bass boost factor (1.0 = none, 2.0 = double)
        lowpass: Lowpass filter cutoff Hz (None = disabled)
        highpass: Highpass filter cutoff Hz (None = disabled)
        smooth: Smoothing window size (0 = disabled)

    Returns:
        Processed audio
    """
    # Apply filters in order
    if highpass and highpass > 0:
        audio = apply_highpass(audio, sample_rate, highpass)

    if lowpass and lowpass > 0:
        audio = apply_lowpass(audio, sample_rate, lowpass)

    if bass_boost > 1.0:
        audio = apply_bass_boost(audio, sample_rate, bass_boost)

    if smooth > 0:
        audio = apply_smooth(audio, smooth)

    return audio


def scanline_video(
    video_path: str,
    direction: str = "horizontal",
    channel_mode: str = "mono",
    sample_rate: int = 44100,
    invert: bool = False,
    bass_boost: float = 1.0,
    lowpass: int = None,
    highpass: int = None,
    smooth: int = 0,
    verbose: bool = False,
    progress_callback=None,
) -> np.ndarray:
    """
    Convert video to audio using scanline sonification.

    Each frame produces a chunk of audio, matched to video duration.

    Args:
        video_path: Path to video file
        direction: 'horizontal' or 'vertical'
        channel_mode: 'mono', 'stereo', or 'rgb'
        sample_rate: Audio sample rate in Hz
        invert: If True, dark pixels become loud
        bass_boost: Bass boost factor (1.0 = none)
        lowpass: Lowpass filter cutoff Hz (None = disabled)
        highpass: Highpass filter cutoff Hz (None = disabled)
        smooth: Smoothing window size (0 = disabled)
        verbose: Print progress info
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Audio samples as np.ndarray
    """
    info = get_video_info(video_path)
    fps = info["fps"]
    frame_count = info["frame_count"]
    video_duration = info["duration"]

    if fps <= 0:
        raise ValueError("Could not determine video FPS")

    samples_per_frame = int(sample_rate / fps)

    audio_chunks = []
    frames = extract_frames(video_path)

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i, frame_count)
        elif verbose and (i % 50 == 0):
            print(f"Processing frame {i + 1}/{frame_count}")

        chunk = scanline_image(frame, direction, channel_mode, invert)
        chunk = resample_audio(chunk, samples_per_frame)
        audio_chunks.append(chunk)

    if progress_callback:
        progress_callback(frame_count, frame_count)

    if not audio_chunks:
        raise ValueError("No frames extracted from video")

    audio = np.concatenate(audio_chunks)

    # Apply audio processing
    audio = process_scanline_audio(
        audio,
        sample_rate,
        bass_boost=bass_boost,
        lowpass=lowpass,
        highpass=highpass,
        smooth=smooth,
    )

    # Ensure exact video duration
    target_samples = int(video_duration * sample_rate)
    if len(audio) != target_samples:
        audio = resample_audio(audio, target_samples)

    return audio
