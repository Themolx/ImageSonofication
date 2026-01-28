"""Spectral sonification - treat image as spectrogram and reconstruct via iSTFT."""

import numpy as np
from scipy import signal
from scipy.ndimage import zoom

from img2sound.io.image import to_grayscale
from img2sound.io.video import extract_frames, get_video_info
from img2sound.core.audio import resample_audio


def _resize_image(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize image to target shape using scipy zoom."""
    zoom_factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def _create_linear_freq_mapping(
    n_bins: int, freq_min: int, freq_max: int, sample_rate: int
) -> np.ndarray:
    """Create linear frequency bin mapping."""
    freqs = np.fft.rfftfreq(2 * (n_bins - 1), 1.0 / sample_rate)
    bin_min = np.searchsorted(freqs, freq_min)
    bin_max = np.searchsorted(freqs, freq_max)
    return np.linspace(bin_min, bin_max, n_bins).astype(int)


def _create_log_freq_mapping(
    n_bins: int, freq_min: int, freq_max: int, sample_rate: int
) -> np.ndarray:
    """Create logarithmic frequency bin mapping (more musical)."""
    freqs = np.fft.rfftfreq(2 * (n_bins - 1), 1.0 / sample_rate)
    log_freqs = np.logspace(np.log10(max(freq_min, 1)), np.log10(freq_max), n_bins)
    indices = np.searchsorted(freqs, log_freqs)
    return np.clip(indices, 0, len(freqs) - 1).astype(int)


def _apply_freq_mapping(
    img: np.ndarray, freq_indices: np.ndarray, n_bins: int
) -> np.ndarray:
    """Map image rows to frequency bins."""
    result = np.zeros((n_bins, img.shape[1]), dtype=np.float32)

    for i, src_row in enumerate(np.linspace(0, img.shape[0] - 1, n_bins).astype(int)):
        result[i, :] = img[src_row, :]

    return result


def _apply_bass_boost(magnitude: np.ndarray, boost_factor: float = 2.0) -> np.ndarray:
    """Apply bass boost - increase magnitude of lower frequencies."""
    n_bins = magnitude.shape[0]
    # Create boost curve: strong at low freqs, tapering to 1.0 at high freqs
    boost_curve = np.linspace(boost_factor, 1.0, n_bins)
    return magnitude * boost_curve[:, np.newaxis]


def _istft(stft_matrix: np.ndarray, hop_length: int, n_fft: int) -> np.ndarray:
    """Inverse Short-Time Fourier Transform."""
    _, frames = stft_matrix.shape
    expected_signal_len = n_fft + hop_length * (frames - 1)

    audio = np.zeros(expected_signal_len, dtype=np.float64)
    window = signal.get_window("hann", n_fft)

    for i in range(frames):
        frame_start = i * hop_length
        frame_spectrum = stft_matrix[:, i]
        frame_time = np.fft.irfft(frame_spectrum, n_fft)
        audio[frame_start : frame_start + n_fft] += frame_time * window

    # Normalize by window overlap
    norm = np.zeros(expected_signal_len)
    for i in range(frames):
        frame_start = i * hop_length
        norm[frame_start : frame_start + n_fft] += window**2

    norm[norm < 1e-8] = 1e-8
    audio = audio / norm

    return audio.astype(np.float32)


def _stft(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Short-Time Fourier Transform."""
    window = signal.get_window("hann", n_fft)
    frames = 1 + (len(audio) - n_fft) // hop_length

    stft_matrix = np.zeros((n_fft // 2 + 1, frames), dtype=np.complex128)

    for i in range(frames):
        frame_start = i * hop_length
        frame = audio[frame_start : frame_start + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(frame)

    return stft_matrix


def griffin_lim(
    magnitude: np.ndarray, n_fft: int, hop_length: int, iterations: int
) -> np.ndarray:
    """
    Griffin-Lim algorithm for phase reconstruction.

    Args:
        magnitude: Magnitude spectrogram
        n_fft: FFT size
        hop_length: Hop between frames
        iterations: Number of iterations

    Returns:
        Reconstructed audio
    """
    phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)

    for _ in range(iterations):
        stft_matrix = magnitude * np.exp(1j * phase)
        audio = _istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)
        new_stft = _stft(audio, n_fft=n_fft, hop_length=hop_length)
        phase = np.angle(new_stft)

    stft_matrix = magnitude * np.exp(1j * phase)
    audio = _istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)

    return audio


def spectral_image(
    image: np.ndarray,
    duration: float = 5.0,
    sample_rate: int = 44100,
    freq_min: int = 30,
    freq_max: int = 10000,
    phase_mode: str = "random",
    griffin_iterations: int = 32,
    log_freq: bool = True,
    bass_boost: float = 2.5,
) -> np.ndarray:
    """
    Treat image as spectrogram and reconstruct audio.

    X-axis = time, Y-axis = frequency (bottom = low, top = high),
    Brightness = magnitude.

    Args:
        image: Input image (will be converted to grayscale)
        duration: Output duration in seconds
        sample_rate: Audio sample rate
        freq_min: Minimum frequency in Hz (default 30 for bass)
        freq_max: Maximum frequency in Hz
        phase_mode: 'random', 'griffin-lim', or 'zero'
        griffin_iterations: Number of iterations for Griffin-Lim
        log_freq: If True, use logarithmic frequency mapping (default True)
        bass_boost: Bass boost factor (1.0 = no boost, 2.5 = strong bass)

    Returns:
        Audio samples as np.ndarray
    """
    img = to_grayscale(image).astype(np.float32)

    val_min = np.min(img)
    val_max = np.max(img)
    val_range = val_max - val_min

    if val_range < 1e-10:
        target_samples = int(duration * sample_rate)
        return np.zeros(target_samples, dtype=np.float32)

    img = (img - val_min) / val_range
    img = np.flipud(img)

    # STFT parameters - larger FFT for better frequency resolution
    n_fft = 4096
    hop_length = 1024
    n_bins = n_fft // 2 + 1

    target_frames = int((duration * sample_rate) / hop_length)
    target_frames = max(target_frames, 2)

    img_resized = _resize_image(img, (n_bins, target_frames))

    if log_freq:
        freq_indices = _create_log_freq_mapping(n_bins, freq_min, freq_max, sample_rate)
    else:
        freq_indices = _create_linear_freq_mapping(n_bins, freq_min, freq_max, sample_rate)

    magnitude = _apply_freq_mapping(img_resized, freq_indices, n_bins)

    # Apply bass boost for fuller sound
    if bass_boost > 1.0:
        magnitude = _apply_bass_boost(magnitude, bass_boost)

    # Higher magnitude scaling for louder output
    magnitude = magnitude * 20.0

    if phase_mode == "griffin-lim":
        audio = griffin_lim(magnitude, n_fft, hop_length, griffin_iterations)
    else:
        if phase_mode == "random":
            phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        elif phase_mode == "zero":
            phase = np.zeros_like(magnitude)
        else:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        stft_matrix = magnitude * np.exp(1j * phase)
        audio = _istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)

    target_samples = int(duration * sample_rate)
    if len(audio) != target_samples:
        audio = resample_audio(audio, target_samples)

    return audio


def spectral_video(
    video_path: str,
    sample_rate: int = 44100,
    freq_min: int = 30,
    freq_max: int = 10000,
    phase_mode: str = "random",
    griffin_iterations: int = 32,
    log_freq: bool = True,
    bass_boost: float = 2.5,
    verbose: bool = False,
    progress_callback=None,
) -> np.ndarray:
    """
    Convert video to audio using spectral sonification.

    Each frame becomes a short spectrogram slice. Output duration
    matches video duration exactly for real-time sync.

    Args:
        video_path: Path to video file
        sample_rate: Audio sample rate
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        phase_mode: 'random', 'griffin-lim', or 'zero'
        griffin_iterations: Number of iterations for Griffin-Lim
        log_freq: If True, use logarithmic frequency mapping
        bass_boost: Bass boost factor
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

    samples_per_frame = int(sample_rate / fps)
    frame_duration = 1.0 / fps

    audio_chunks = []
    frames = extract_frames(video_path)

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i, frame_count)
        elif verbose and (i % 50 == 0):
            print(f"Processing frame {i + 1}/{frame_count}")

        chunk = spectral_image(
            frame,
            duration=frame_duration,
            sample_rate=sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
            phase_mode=phase_mode,
            griffin_iterations=griffin_iterations,
            log_freq=log_freq,
            bass_boost=bass_boost,
        )
        if len(chunk) != samples_per_frame:
            chunk = resample_audio(chunk, samples_per_frame)
        audio_chunks.append(chunk)

    if progress_callback:
        progress_callback(frame_count, frame_count)

    if not audio_chunks:
        raise ValueError("No frames extracted from video")

    audio = np.concatenate(audio_chunks)

    target_samples = int(video_duration * sample_rate)
    if len(audio) != target_samples:
        audio = resample_audio(audio, target_samples)

    return audio
