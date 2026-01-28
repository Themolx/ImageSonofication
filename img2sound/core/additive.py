"""Additive sonification - each image row becomes a sine oscillator."""

import numpy as np

from img2sound.io.image import to_grayscale
from img2sound.io.video import extract_frames, get_video_info
from img2sound.core.audio import resample_audio


def additive_image(
    image: np.ndarray,
    duration: float = 5.0,
    sample_rate: int = 44100,
    freq_min: int = 40,
    freq_max: int = 4000,
    log_freq: bool = True,
    n_oscillators: int = 64,
) -> np.ndarray:
    """
    Convert image to audio using additive synthesis.

    Each row of the image becomes a sine wave oscillator.
    Y-axis = frequency (bottom = low, top = high)
    X-axis = time
    Brightness = amplitude of that frequency at that moment

    This creates a very direct visual-to-audio mapping:
    - Horizontal lines = sustained tones
    - Vertical lines = clicks/transients
    - Diagonal lines = pitch sweeps
    - Bright areas = loud frequencies

    Args:
        image: Input image (will be converted to grayscale)
        duration: Output duration in seconds
        sample_rate: Audio sample rate
        freq_min: Minimum frequency in Hz (bottom of image)
        freq_max: Maximum frequency in Hz (top of image)
        log_freq: Use logarithmic frequency spacing (more musical)
        n_oscillators: Number of frequency bands/oscillators

    Returns:
        Audio samples as np.ndarray
    """
    # Convert to grayscale and normalize
    img = to_grayscale(image).astype(np.float32)

    # Auto-scale brightness
    val_min = np.min(img)
    val_max = np.max(img)
    val_range = val_max - val_min

    if val_range < 1e-10:
        return np.zeros(int(duration * sample_rate), dtype=np.float32)

    img = (img - val_min) / val_range  # [0, 1]

    # Flip so bottom = low frequency
    img = np.flipud(img)

    height, width = img.shape
    n_samples = int(duration * sample_rate)

    # Generate frequency array
    if log_freq:
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_oscillators)
    else:
        frequencies = np.linspace(freq_min, freq_max, n_oscillators)

    # Resample image to match oscillator count and time samples
    # Height -> n_oscillators, Width -> time frames
    from scipy.ndimage import zoom

    # Calculate zoom factors
    zoom_h = n_oscillators / height
    zoom_w = n_samples / width

    # Resample amplitude envelope for each oscillator
    amp_envelope = zoom(img, (zoom_h, zoom_w), order=1)  # (n_oscillators, n_samples)

    # Generate audio using vectorized synthesis
    t = np.arange(n_samples) / sample_rate  # Time array

    # Build audio by summing oscillators
    audio = np.zeros(n_samples, dtype=np.float64)

    for i, freq in enumerate(frequencies):
        # Generate sine wave
        sine = np.sin(2 * np.pi * freq * t)
        # Apply amplitude envelope for this frequency
        audio += sine * amp_envelope[i, :]

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio.astype(np.float32)


def additive_video(
    video_path: str,
    sample_rate: int = 44100,
    freq_min: int = 40,
    freq_max: int = 4000,
    log_freq: bool = True,
    n_oscillators: int = 64,
    verbose: bool = False,
    progress_callback=None,
) -> np.ndarray:
    """
    Convert video to audio using additive synthesis.

    Each frame's brightness distribution controls oscillator amplitudes.
    Creates direct visual-to-audio sync: what you see = what you hear.

    Args:
        video_path: Path to video file
        sample_rate: Audio sample rate
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        log_freq: Use logarithmic frequency spacing
        n_oscillators: Number of frequency bands
        verbose: Print progress info
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Audio samples as np.ndarray (duration matches video)
    """
    info = get_video_info(video_path)
    fps = info["fps"]
    frame_count = info["frame_count"]
    video_duration = info["duration"]

    if fps <= 0:
        raise ValueError("Could not determine video FPS")

    # Generate frequency array once
    if log_freq:
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_oscillators)
    else:
        frequencies = np.linspace(freq_min, freq_max, n_oscillators)

    # Calculate samples per frame
    samples_per_frame = int(sample_rate / fps)
    total_samples = int(video_duration * sample_rate)

    # Pre-allocate output
    audio = np.zeros(total_samples, dtype=np.float64)

    # Time array for one frame
    t_frame = np.arange(samples_per_frame) / sample_rate

    # Pre-compute sine waves for each frequency (one frame worth)
    sines = np.array([np.sin(2 * np.pi * f * t_frame) for f in frequencies])  # (n_osc, samples_per_frame)

    frames = extract_frames(video_path)

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i, frame_count)
        elif verbose and (i % 50 == 0):
            print(f"Processing frame {i + 1}/{frame_count}")

        # Convert frame to amplitude envelope
        img = to_grayscale(frame).astype(np.float32)

        # Normalize
        val_min = np.min(img)
        val_max = np.max(img)
        val_range = val_max - val_min

        if val_range > 1e-10:
            img = (img - val_min) / val_range
        else:
            img = np.zeros_like(img)

        # Flip and resize to oscillator count
        img = np.flipud(img)

        # Average each horizontal band to get amplitude per oscillator
        band_height = img.shape[0] / n_oscillators
        amplitudes = np.zeros(n_oscillators)

        for j in range(n_oscillators):
            y_start = int(j * band_height)
            y_end = int((j + 1) * band_height)
            amplitudes[j] = np.mean(img[y_start:y_end, :])

        # Generate audio for this frame
        frame_audio = np.sum(sines * amplitudes[:, np.newaxis], axis=0)

        # Place in output
        start_sample = i * samples_per_frame
        end_sample = min(start_sample + samples_per_frame, total_samples)
        actual_len = end_sample - start_sample

        if actual_len > 0:
            audio[start_sample:end_sample] = frame_audio[:actual_len]

    if progress_callback:
        progress_callback(frame_count, frame_count)

    # Final normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Ensure exact duration
    if len(audio) != total_samples:
        audio = resample_audio(audio, total_samples)

    return audio.astype(np.float32)
