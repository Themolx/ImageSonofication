"""Audio utilities for normalization, WAV writing, and resampling."""

import numpy as np
from scipy.io import wavfile
from scipy import signal


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to peak amplitude.

    Args:
        audio: Audio samples as numpy array
        peak: Target peak amplitude (0.0 to 1.0)

    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (peak / max_val)
    return audio


def auto_scale_audio(data: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Automatically analyze min/max values and scale to full audio range.

    This analyzes the actual value range in the data and maps it
    intelligently to [-peak, +peak]. Works with any input range
    (0-255, 0-1, arbitrary values).

    Args:
        data: Input data as numpy array (any range)
        peak: Target peak amplitude (0.0 to 1.0)

    Returns:
        Scaled audio array in range [-peak, +peak]
    """
    data = data.astype(np.float64)

    # Find actual min/max
    val_min = np.min(data)
    val_max = np.max(data)
    val_range = val_max - val_min

    if val_range < 1e-10:
        # Constant value - return silence
        return np.zeros_like(data, dtype=np.float32)

    # Map [val_min, val_max] -> [-peak, +peak]
    # First normalize to [0, 1], then scale to [-peak, peak]
    normalized = (data - val_min) / val_range  # [0, 1]
    audio = (normalized * 2.0 - 1.0) * peak    # [-peak, peak]

    return audio.astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sample_rate: int, bit_depth: int = 16) -> None:
    """
    Write audio to WAV file.

    Args:
        path: Output file path
        audio: Audio samples as numpy array (float, -1.0 to 1.0)
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (16 or 32)
    """
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    if bit_depth == 16:
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(path, sample_rate, audio_int)
    elif bit_depth == 32:
        wavfile.write(path, sample_rate, audio.astype(np.float32))
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 16 or 32.")


def resample_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample audio to target number of samples.

    Args:
        audio: Audio samples as numpy array
        target_length: Target number of samples

    Returns:
        Resampled audio array
    """
    if len(audio) == target_length:
        return audio

    if audio.ndim == 1:
        return signal.resample(audio, target_length)
    else:
        # Handle stereo/multi-channel
        resampled = np.zeros((target_length, audio.shape[1]), dtype=audio.dtype)
        for ch in range(audio.shape[1]):
            resampled[:, ch] = signal.resample(audio[:, ch], target_length)
        return resampled


def apply_bass_boost(audio: np.ndarray, sample_rate: int, boost: float = 2.0, cutoff: int = 200) -> np.ndarray:
    """
    Apply bass boost by amplifying low frequencies.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        boost: Boost factor (1.0 = no boost, 2.0 = double bass)
        cutoff: Frequency below which to boost (Hz)

    Returns:
        Bass-boosted audio
    """
    if boost <= 1.0:
        return audio

    # Design lowpass filter to extract bass
    nyquist = sample_rate / 2
    normalized_cutoff = min(cutoff / nyquist, 0.99)

    b, a = signal.butter(2, normalized_cutoff, btype='low')

    # Handle mono vs stereo
    if audio.ndim == 1:
        bass = signal.filtfilt(b, a, audio)
        return audio + bass * (boost - 1.0)
    else:
        result = audio.copy()
        for ch in range(audio.shape[1]):
            bass = signal.filtfilt(b, a, audio[:, ch])
            result[:, ch] = audio[:, ch] + bass * (boost - 1.0)
        return result


def apply_lowpass(audio: np.ndarray, sample_rate: int, cutoff: int) -> np.ndarray:
    """
    Apply lowpass filter.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz

    Returns:
        Filtered audio
    """
    nyquist = sample_rate / 2
    normalized_cutoff = min(cutoff / nyquist, 0.99)

    b, a = signal.butter(4, normalized_cutoff, btype='low')

    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio).astype(np.float32)
    else:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = signal.filtfilt(b, a, audio[:, ch])
        return result.astype(np.float32)


def apply_highpass(audio: np.ndarray, sample_rate: int, cutoff: int) -> np.ndarray:
    """
    Apply highpass filter.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz

    Returns:
        Filtered audio
    """
    nyquist = sample_rate / 2
    normalized_cutoff = max(cutoff / nyquist, 0.01)

    b, a = signal.butter(4, normalized_cutoff, btype='high')

    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio).astype(np.float32)
    else:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = signal.filtfilt(b, a, audio[:, ch])
        return result.astype(np.float32)


def apply_smooth(audio: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth audio to reduce harshness.

    Args:
        audio: Audio samples
        window_size: Smoothing window size (odd number)

    Returns:
        Smoothed audio
    """
    if window_size <= 1:
        return audio

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size) / window_size

    if audio.ndim == 1:
        return np.convolve(audio, kernel, mode='same').astype(np.float32)
    else:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = np.convolve(audio[:, ch], kernel, mode='same')
        return result.astype(np.float32)


def crossfade_concatenate(chunks: list, crossfade_samples: int = 64) -> np.ndarray:
    """
    Concatenate audio chunks with crossfade to avoid clicks.

    Args:
        chunks: List of audio arrays
        crossfade_samples: Number of samples for crossfade

    Returns:
        Concatenated audio array
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0]

    # Calculate total length
    total_length = sum(len(c) for c in chunks) - crossfade_samples * (len(chunks) - 1)

    # Determine output shape
    if chunks[0].ndim == 1:
        result = np.zeros(total_length, dtype=np.float32)
    else:
        result = np.zeros((total_length, chunks[0].shape[1]), dtype=np.float32)

    # Crossfade window
    fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        if i == 0:
            # First chunk: no fade in
            if chunk.ndim == 1:
                result[pos:pos + chunk_len - crossfade_samples] = chunk[:-crossfade_samples]
                result[pos + chunk_len - crossfade_samples:pos + chunk_len] = chunk[-crossfade_samples:] * fade_out
            else:
                result[pos:pos + chunk_len - crossfade_samples] = chunk[:-crossfade_samples]
                result[pos + chunk_len - crossfade_samples:pos + chunk_len] = chunk[-crossfade_samples:] * fade_out[:, np.newaxis]
            pos += chunk_len - crossfade_samples
        elif i == len(chunks) - 1:
            # Last chunk: no fade out
            if chunk.ndim == 1:
                result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in
                result[pos + crossfade_samples:pos + chunk_len] = chunk[crossfade_samples:]
            else:
                result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in[:, np.newaxis]
                result[pos + crossfade_samples:pos + chunk_len] = chunk[crossfade_samples:]
        else:
            # Middle chunks: both fade in and fade out
            if chunk.ndim == 1:
                result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in
                result[pos + crossfade_samples:pos + chunk_len - crossfade_samples] = chunk[crossfade_samples:-crossfade_samples]
                result[pos + chunk_len - crossfade_samples:pos + chunk_len] = chunk[-crossfade_samples:] * fade_out
            else:
                result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in[:, np.newaxis]
                result[pos + crossfade_samples:pos + chunk_len - crossfade_samples] = chunk[crossfade_samples:-crossfade_samples]
                result[pos + chunk_len - crossfade_samples:pos + chunk_len] = chunk[-crossfade_samples:] * fade_out[:, np.newaxis]
            pos += chunk_len - crossfade_samples

    return result
