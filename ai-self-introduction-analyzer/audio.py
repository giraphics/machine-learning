"""Extract audio from video using ffmpeg."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_ffmpeg() -> str:
    """Return path to ffmpeg executable, or raise FileNotFoundError with install hint."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg and add it to your PATH.\n"
        "  Windows: winget install ffmpeg  or  choco install ffmpeg\n"
        "  Or download from https://ffmpeg.org/download.html"
    )


def extract_audio(video_path: str | Path) -> Path:
    """
    Extract audio from a video file to a temporary WAV file.
    Returns the path to the temporary WAV file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    suffix = video_path.suffix.lower()
    if suffix not in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4a", ".mp3", ".wav"):
        print(f"Warning: unusual extension {suffix}, attempting extraction anyway.", file=sys.stderr)

    ffmpeg_exe = _find_ffmpeg()

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_path = Path(wav_path)

    # mono 16kHz is ideal for Whisper; ffmpeg does the conversion
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        wav_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {result.stderr or result.stdout}")

    return wav_path
