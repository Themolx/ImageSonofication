"""Common utilities and constants."""

import re
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

    # Find next version number
    version = 1
    pattern = re.compile(rf"^{re.escape(name_base)}_v(\d{{3}}){re.escape(extension)}$")

    for existing_file in date_folder.iterdir():
        match = pattern.match(existing_file.name)
        if match:
            existing_version = int(match.group(1))
            version = max(version, existing_version + 1)

    # Generate filename
    filename = f"{name_base}_v{version:03d}{extension}"

    return date_folder / filename
