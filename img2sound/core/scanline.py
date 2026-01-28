"""Scanline sonification - spatial mapping of brightness to amplitude."""

import numpy as np
from typing import Generator

from img2sound.io.image import to_grayscale
from img2sound.io.video import extract_frames, get_video_info
from img2sound.core.audio import resample_audio, auto_scale_audio


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

    Automatically analyzes the actual brightness range and maps it
    to full audio range for optimal dynamics.

    Args:
        image: RGB or grayscale image array (H, W, C) or (H, W)
        direction: 'horizontal' (L->R) or 'vertical' (T->B)
        channel_mode: 'mono', 'stereo', or 'rgb'
        invert: If True, dark pixels become loud

    Returns:
        Audio samples as np.ndarray, shape (N,) for mono, (N, 2) for stereo
    """
    # Ensure float for processing
    img = image.astype(np.float32)

    # Convert to appropriate format based on channel mode
    if channel_mode == "mono":
        img = to_grayscale(img)  # (H, W)
    elif img.ndim == 2:
        # If grayscale input but stereo/rgb mode requested, expand
        img = np.stack([img, img, img], axis=-1)

    # Transpose if vertical scan
    if direction == "vertical":
        if img.ndim == 2:
            img = np.transpose(img, (1, 0))  # (H, W) -> (W, H)
        else:
            img = np.transpose(img, (1, 0, 2))  # (H, W, C) -> (W, H, C)

    # Flatten column by column for detail
    if channel_mode == "mono":
        # Each column's pixels become sequential samples
        audio = img.T.flatten()  # (W, H) flattened -> (W*H,)
    elif channel_mode == "stereo":
        # Red channel -> left, Blue channel -> right
        left = img[:, :, 0].T.flatten()  # Red
        right = img[:, :, 2].T.flatten()  # Blue
        audio = np.stack([left, right], axis=1)
    elif channel_mode == "rgb":
        # All three channels interleaved or as separate outputs
        # Here we interleave: R, G, B, R, G, B, ...
        r = img[:, :, 0].T.flatten()
        g = img[:, :, 1].T.flatten()
        b = img[:, :, 2].T.flatten()
        audio = np.stack([r, g, b], axis=1)
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    # Auto-scale: analyze actual min/max and map to full audio range
    audio = auto_scale_audio(audio)

    # Invert if requested
    if invert:
        audio = -audio

    return audio.astype(np.float32)


def scanline_video(
    video_path: str,
    direction: str = "horizontal",
    channel_mode: str = "mono",
    sample_rate: int = 44100,
    invert: bool = False,
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

    # Ensure exact video duration
    target_samples = int(video_duration * sample_rate)
    if len(audio) != target_samples:
        audio = resample_audio(audio, target_samples)

    return audio
