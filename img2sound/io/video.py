"""Video frame extraction utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator

from img2sound.utils.helpers import VIDEO_EXTENSIONS, UnsupportedFormatError, ProcessingError


def extract_frames(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Yield frames from video as RGB numpy arrays.

    Args:
        video_path: Path to video file

    Yields:
        Frames as numpy arrays (H, W, 3) in RGB format

    Raises:
        UnsupportedFormatError: If file extension not supported
        ProcessingError: If video cannot be opened
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    ext = path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported video format: {ext}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ProcessingError(f"Failed to open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info:
        - fps: Frames per second
        - frame_count: Total number of frames
        - width: Frame width
        - height: Frame height
        - duration: Video duration in seconds

    Raises:
        ProcessingError: If video cannot be opened
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ProcessingError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
        }
    finally:
        cap.release()


def get_video_fps(video_path: str) -> float:
    """
    Get video frames per second.

    Args:
        video_path: Path to video file

    Returns:
        FPS as float
    """
    return get_video_info(video_path)["fps"]
