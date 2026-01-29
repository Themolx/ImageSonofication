"""Common utilities and constants."""

import re
import fcntl
import time
from datetime import datetime
from pathlib import Path


# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


class Img2SoundError(Exception):
    """Base exception for img2sound."""

    pass


class UnsupportedFormatError(Img2SoundError):
    """Raised when input format is not supported."""

    pass


class ProcessingError(Img2SoundError):
    """Raised when processing fails."""

    pass


def get_input_type(path: str) -> str:
    """
    Determine input type from file extension.

    Args:
        path: Path to input file

    Returns:
        'image' or 'video'

    Raises:
        UnsupportedFormatError: If extension not recognized
    """
    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    else:
        raise UnsupportedFormatError(
            f"Unsupported file type: {ext}. "
            f"Supported images: {', '.join(sorted(IMAGE_EXTENSIONS))}. "
            f"Supported videos: {', '.join(sorted(VIDEO_EXTENSIONS))}."
        )


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1:23.45" or "0:05.12")
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def format_filesize(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def generate_output_path(input_path: str, mode: str = None, output_dir: str = None, extension: str = ".wav") -> Path:
    """
    Generate a versioned output path in date-based folder structure.

    Creates paths like: YYMMDD/YYMMDD_<filename>_<mode>_v001.wav
    Auto-increments version number if file exists.
    Uses file locking to prevent race conditions when running multiple instances.

    Args:
        input_path: Path to input file (used to extract base name)
        mode: Processing mode name (scanline, spectral, additive)
        output_dir: Base output directory (default: current directory)
        extension: Output file extension (default: .wav)

    Returns:
        Path object for the versioned output file
    """
    input_path = Path(input_path)
    base_name = input_path.stem  # filename without extension

    # Get date string
    date_str = datetime.now().strftime("%y%m%d")

    # Determine base output directory
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path.cwd()

    # Create date folder
    date_folder = base_dir / date_str
    date_folder.mkdir(parents=True, exist_ok=True)

    # Build name pattern with optional mode
    if mode:
        name_base = f"{date_str}_{base_name}_{mode}"
    else:
        name_base = f"{date_str}_{base_name}"

    # Use a lock file to prevent race conditions when running multiple instances
    lock_file = date_folder / ".img2sound.lock"

    with open(lock_file, "w") as lf:
        # Acquire exclusive lock (blocks until available)
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)

        try:
            # Find next version number by checking both .wav and .nfo files
            version = 1
            # Match any extension to catch all versioned files (wav, nfo, etc.)
            pattern = re.compile(rf"^{re.escape(name_base)}_v(\d{{3}})\.[a-zA-Z0-9]+$")

            for existing_file in date_folder.iterdir():
                # Skip directories and lock file
                if not existing_file.is_file() or existing_file.name.startswith("."):
                    continue
                match = pattern.match(existing_file.name)
                if match:
                    existing_version = int(match.group(1))
                    version = max(version, existing_version + 1)

            # Generate filename
            filename = f"{name_base}_v{version:03d}{extension}"
            output_path = date_folder / filename

            # Create a placeholder file to reserve this version number
            output_path.touch()

        finally:
            # Release lock
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    return output_path


def write_nfo(wav_path: str, settings: dict) -> Path:
    """
    Write NFO file with render settings alongside WAV file.

    Args:
        wav_path: Path to the WAV file
        settings: Dictionary of settings used for the render

    Returns:
        Path to the NFO file
    """
    wav_path = Path(wav_path)
    nfo_path = wav_path.with_suffix(".nfo")

    lines = [
        f"img2sound render settings",
        f"========================",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]

    # Add all settings
    for key, value in settings.items():
        if value is not None:
            # Format the key nicely
            display_key = key.replace("_", " ").title()
            lines.append(f"{display_key}: {value}")

    # Write the file
    nfo_path.write_text("\n".join(lines))

    return nfo_path
