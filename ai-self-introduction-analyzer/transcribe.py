"""Transcribe audio using OpenAI Whisper."""

from pathlib import Path


def transcribe(audio_path: str | Path, model: str = "base") -> str:
    """
    Transcribe audio file to text using Whisper.
    model: "tiny", "base", "small", "medium", "large" or "large-v3"
    """
    import whisper

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_obj = whisper.load_model(model)
    result = model_obj.transcribe(str(audio_path), fp16=False)
    return (result.get("text") or "").strip()
