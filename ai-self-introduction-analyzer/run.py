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

# Saved custom prompt for Ollama (prioritized over default when present)
def _saved_prompt_path() -> Path:
    return Path(__file__).resolve().parent / "saved_ollama_prompt.txt"

# Placeholder that must appear in custom prompts so we can inject the transcript
TRANSCRIPT_PLACEHOLDER = "{transcript}"


def _read_instructions(path: Path) -> str:
    """Read instructions from a Markdown or plain text file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instructions file not found: {path}")
    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode instructions file: {path}")


def _build_prompt_from_instructions(instructions: str) -> str:
    """Build a polished, unambiguous Ollama prompt template from user instructions."""
    return f"""You are an expert coach for self-introductions and public speaking. Your task is to analyze transcripts and give structured, actionable feedback.

## Criteria and instructions to follow

{instructions}

## Transcript to analyze

---
{TRANSCRIPT_PLACEHOLDER}
---

## Required output format

Provide your analysis in this exact format so it can be parsed reliably:

Feedback:
- [2–4 bullet points: structure, pacing, clarity, standout positives]
- Repeated filler words: [list any, e.g. "um", "uh", "like", or "none" if minimal]
- Clarity score: X/10
- Fluency score: X/10
- Confidence score: X/10
- Structure score: X/10
- Vocabulary score: X/10
- Suggestion: [one concrete, specific tip]

Be concise and consistent. Use only the format above."""


def _load_prompt_for_ollama(instructions_path: Path | None) -> tuple[str | None, bool]:
    """
    Return (prompt_template, from_instructions).
    Priority: saved file > generated from --instructions > (None, False).
    from_instructions is True only when we built from --instructions (no saved file yet).
    """
    saved = _saved_prompt_path()
    if saved.exists():
        for enc in ("utf-8", "utf-8-sig", "utf-16"):
            try:
                content = saved.read_text(encoding=enc).strip()
                if TRANSCRIPT_PLACEHOLDER not in content:
                    content = content + "\n\n## Transcript to analyze\n\n---\n" + TRANSCRIPT_PLACEHOLDER + "\n---"
                return (content, False)
            except UnicodeDecodeError:
                continue
    if instructions_path and instructions_path.exists():
        return (_build_prompt_from_instructions(_read_instructions(instructions_path)), True)
    return (None, False)


def _show_prompt_and_edit_if_requested(prompt_template: str) -> str:
    """
    Show the prompt to the user and ask if they want to modify it.
    If yes, open in editor and return edited content. Otherwise return as-is.
    Saves the final prompt to saved_ollama_prompt.txt.
    """
    print("\n" + "=" * 60 + "\nGenerated prompt for Ollama:\n" + "=" * 60)
    print(prompt_template)
    print("=" * 60)
    while True:
        reply = input("\nModify this prompt? (yes/y / no/n): ").strip().lower()
        if reply in ("yes", "y"):
            prompt_template = _edit_prompt_in_editor(prompt_template)
            break
        if reply in ("no", "n"):
            break
        print("Please answer yes/y or no/n.")
    saved = _saved_prompt_path()
    saved.write_text(prompt_template, encoding="utf-8")
    print(f"Prompt saved to {saved}")
    return prompt_template


def _edit_prompt_in_editor(content: str) -> str:
    """Write content to a temp file, open in $EDITOR, read back."""
    import subprocess
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="ollama_prompt_")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "notepad"))
        subprocess.run([editor, path], check=False)
        return Path(path).read_text(encoding="utf-8").strip()
    finally:
        Path(path).unlink(missing_ok=True)

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
    prompt_template: str | None = None,
    *,
    quiet: bool = False,
) -> tuple[str, str]:
    """
    Run extract → transcribe → analyze on one video. Returns (transcript, feedback).
    If quiet is False, prints progress and final transcript/feedback.
    prompt_template is used for Ollama when set (must contain {transcript}).
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
        feedback = analyze_transcript(
            transcript,
            llm_model=llm_model,
            prompt_template=prompt_template,
        )
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


def _write_readable_fallback(
    output_dir: Path, stem: str, video_name: str, transcript: str, feedback: str
) -> None:
    """Write a single readable file when JSON/CSV format is not achievable."""
    path = output_dir / f"{stem}_feedback_readable.txt"
    content = (
        f"Video: {video_name}\n\n"
        f"{'='*60}\nTranscript\n{'='*60}\n\n{transcript}\n\n"
        f"{'='*60}\nFeedback (expected outcomes)\n{'='*60}\n\n{feedback}\n"
    )
    path.write_text(content, encoding="utf-8")


def run_folder(
    folder_path: str | Path,
    output_dir: str | Path,
    whisper_model: str = "base",
    llm_model: str | None = None,
    prompt_template: str | None = None,
) -> None:
    """Process all videos in folder; write JSON and CSV (+ per-file .txt) to output_dir. Fallback to readable file if parsing fails."""
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
                prompt_template=prompt_template,
                quiet=True,
            )
            out = {
                "video": name,
                "transcript": transcript,
                "feedback": feedback,
            }
            out_path = output_dir / f"{video_path.stem}.json"
            scores = _parse_feedback_scores(feedback)
            format_achievable = any(scores.values()) or "score" in feedback.lower()
            json_ok = True
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"  -> {out_path.name}")
            except Exception as e:
                json_ok = False
                print(f"  (JSON write failed: {e})", file=sys.stderr)
                _write_readable_fallback(output_dir, video_path.stem, name, transcript, feedback)
                print(f"  -> {video_path.stem}_feedback_readable.txt")
            if json_ok and not format_achievable and feedback.strip():
                _write_readable_fallback(output_dir, video_path.stem, name, transcript, feedback)
                print(f"  -> {video_path.stem}_feedback_readable.txt (format not matched)")

            if json_ok:
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
    parser.add_argument(
        "--instructions",
        type=Path,
        metavar="FILE",
        default=None,
        help="Path to a Markdown or text file with custom evaluation criteria. Used to build the Ollama prompt; prompt is shown for confirmation and saved for future runs.",
    )
    args = parser.parse_args()

    prompt_template = None
    if args.llm_model:
        template, from_instructions = _load_prompt_for_ollama(args.instructions)
        if template is not None:
            if from_instructions:
                try:
                    prompt_template = _show_prompt_and_edit_if_requested(template)
                except FileNotFoundError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                prompt_template = template

    if args.folder is not None:
        run_folder(
            args.folder,
            args.output,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model or None,
            prompt_template=prompt_template,
        )
        return
    if args.video is None:
        parser.error("Provide a video path, or use --folder to process a directory.")
    try:
        run_pipeline(
            args.video,
            whisper_model=args.whisper_model,
            llm_model=args.llm_model or None,
            prompt_template=prompt_template,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
