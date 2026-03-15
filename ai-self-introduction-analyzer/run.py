#!/usr/bin/env python3
"""
AI self-introduction analyzer.

Pipeline: video → audio extraction → Whisper transcription → LLM analysis → feedback.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Video (and audio) extensions when scanning a folder
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}

# Regex to parse "X score: N/10" and "Suggestion: ..." from feedback text
SCORE_PATTERNS = [
    ("clarity_score", re.compile(r"Clarity score:\s*([\d.]+)/10", re.I)),
    ("fluency_score", re.compile(r"Fluency score:\s*([\d.]+)/10", re.I)),
    ("confidence_score", re.compile(r"Confidence score:\s*([\d.]+)/10", re.I)),
    ("structure_score", re.compile(r"Structure score:\s*([\d.]+)/10", re.I)),
    ("vocabulary_score", re.compile(r"Vocabulary score:\s*([\d.]+)/10", re.I)),
]
SUGGESTION_PATTERN = re.compile(r"Suggestion:\s*(.+?)(?=\n|$)", re.I | re.S)


def _parse_feedback_scores(feedback: str) -> dict[str, str]:
    """Extract score fields and suggestion from feedback text for CSV."""
    row = {}
    for key, pattern in SCORE_PATTERNS:
        m = pattern.search(feedback)
        row[key] = m.group(1).strip() if m else ""
    m = SUGGESTION_PATTERN.search(feedback)
    row["suggestion"] = m.group(1).strip() if m else ""
    return row

# Load .env if present (try encodings so Windows UTF-16 or BOM .env works)
def _load_dotenv_safe() -> None:
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()  # may find .env in cwd or parents
        except (ImportError, UnicodeDecodeError):
            pass
        return
    content = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            with open(dotenv_path, encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        return  # could not decode .env
    # Parse simple KEY=VALUE and set os.environ (handles any encoding)
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"')
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("\\'", "'")
            if key not in os.environ:
                os.environ[key] = value


try:
    _load_dotenv_safe()
except Exception:
    pass

from audio import extract_audio
from transcribe import transcribe
from analyze import analyze_transcript


def run_pipeline(
    video_path: str | Path,
    whisper_model: str = "base",
    llm_model: str | None = None,
    *,
    quiet: bool = False,
) -> tuple[str, str]:
    """
    Run extract → transcribe → analyze on one video. Returns (transcript, feedback).
    If quiet is False, prints progress and final transcript/feedback.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    if not quiet:
        print("Extracting audio...")
    wav_path = extract_audio(video_path)
    try:
        if not quiet:
            print("Transcribing with Whisper...")
        transcript = transcribe(wav_path, model=whisper_model)
        if not transcript:
            raise ValueError("No speech detected.")
        if not quiet:
            if llm_model:
                print(f"Analyzing with Ollama ({llm_model})...")
            else:
                print("Analyzing with LLM...")
        feedback = analyze_transcript(transcript, llm_model=llm_model)
    finally:
        wav_path.unlink(missing_ok=True)

    if not quiet:
        print()
        print("Transcript:")
        print(transcript)
        print()
        print("Feedback:")
        print(feedback)
    return transcript, feedback


def run_folder(
    folder_path: str | Path,
    output_dir: str | Path,
    whisper_model: str = "base",
    llm_model: str | None = None,
) -> None:
    """Process all videos in folder; write JSON and CSV (+ per-file .txt) to output_dir."""
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    if not folder_path.is_dir():
        print(f"Error: not a directory: {folder_path}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No video/audio files found in {folder_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(videos)} file(s). Writing results to {output_dir.resolve()}")

    csv_rows: list[dict[str, str]] = []
    for i, video_path in enumerate(videos, 1):
        name = video_path.name
        print(f"[{i}/{len(videos)}] {name}")
        try:
            transcript, feedback = run_pipeline(
                video_path,
                whisper_model=whisper_model,
                llm_model=llm_model,
                quiet=True,
            )
            out = {
                "video": name,
                "transcript": transcript,
                "feedback": feedback,
            }
            out_path = output_dir / f"{video_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"  -> {out_path.name}")

            scores = _parse_feedback_scores(feedback)
            csv_rows.append({
                "video": name,
                "transcript": transcript,
                "feedback": feedback,
                **scores,
            })
            txt_path = output_dir / f"{video_path.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Video: {name}\n\nTranscript:\n{transcript}\n\nFeedback:\n{feedback}\n")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
    if csv_rows:
        csv_path = output_dir / "results.csv"
        fieldnames = [
            "video", "transcript", "feedback",
            "clarity_score", "fluency_score", "confidence_score",
            "structure_score", "vocabulary_score", "suggestion",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  -> {csv_path.name}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a self-introduction video: extract audio, transcribe, get speaking feedback."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a single video file (e.g. .mp4, .mov). Omit when using --folder.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        metavar="DIR",
        default=None,
        help="Process all videos in this folder; write JSON feedback to --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        default=Path("output"),
        help="Output directory for JSON results when using --folder (default: output).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        default=None,
        help="Use local Ollama model (e.g. llama3, mistral). If omitted, uses OpenAI (requires OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    if args.folder is not None:
        run_folder(
            args.folder,
            args.output,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model or None,
        )
        return
    if args.video is None:
        parser.error("Provide a video path, or use --folder to process a directory.")
    try:
        run_pipeline(
            args.video,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model or None,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
