# AI self-introduction analyzer

Pipeline: **video → audio extraction → Whisper transcription → LLM analysis → speaking/confidence feedback**.

## What it analyzes

- Transcript quality, filler words, pace, sentence clarity, grammar
- Confidence indicators
- Scores: clarity, fluency, confidence, structure, vocabulary
- One concrete suggestion (e.g. “Pause more after key sentences”)

## Prerequisites

- **Python 3.10+**
- **ffmpeg** on your PATH (for audio extraction from video)
- **LLM:** either an **OpenAI API key** (for cloud) or **Ollama** (for local, no cost)

### Installing ffmpeg

If you see “ffmpeg not found”, install it and ensure it’s on your PATH:

- **Windows:** `winget install ffmpeg` or `choco install ffmpeg`, or [download](https://ffmpeg.org/download.html#build-windows) and add the `bin` folder to PATH.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg` (Debian/Ubuntu) or your distro’s package manager.

## Setup

**If you use OpenAI (cloud):** create a `.env` file in the project directory with your API key:

```bash
# Example (Unix/macOS)
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

**Windows (PowerShell):** Avoid `echo KEY=value > .env` — that creates a UTF-16 file and can break loading. Use one of:

- **Notepad:** Create `.env`, add `OPENAI_API_KEY=sk-...`, then **Save As** → Encoding: **UTF-8**.
- **PowerShell (UTF-8):**  
  `"OPENAI_API_KEY=sk-your-key-here" | Out-File -FilePath .env -Encoding utf8`

**If you use Ollama (local, no API key):** install [Ollama](https://ollama.com), start it, and pull a model:

```bash
ollama pull llama3
# or: ollama pull mistral
```

No `.env` or API key needed when using `--llm-model`.

------

# Option A: Using uv (project environment)

[uv](https://docs.astral.sh/uv/) installs dependencies and runs the script without manually activating a venv.

1. Install uv:
   https://docs.astral.sh/uv/getting-started/installation/
2. From the project directory, sync the environment (creates `.venv` and installs deps):

```bash
uv sync
```

1. Run the analyzer:

```bash
uv run python run.py path/to/your_intro.mp4
```

or

```bash
uv run run.py path/to/your_intro.mp4
```

`uv run` automatically uses the project virtual environment.

------

# Option B: Using pip (project virtualenv)

1. Create and activate a virtualenv:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the analyzer:

```bash
python run.py path/to/your_intro.mp4
```

------

# Option C: Using a Global AI Environment (recommended for multiple AI tools)

If you already maintain a **shared AI development environment** (for example `ai-lab`) that contains common AI libraries like:

- torch
- transformers
- whisper
- opencv

you can reuse that environment instead of creating `.venv` for every project.

This avoids duplicated environments when working with multiple AI experiments.

## Activate the global environment

### Windows

```bash
C:\Users\parminder_local\ai-lab\Scripts\activate

or 

C:\Users\parminder_local\ai-lab\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source ~/ai-lab/bin/activate
```

Your terminal should now show:

```
(ai-lab)
```

## Install project dependencies (if needed)

```bash
pip install -r requirements.txt
```

or

```bash
uv pip install -r requirements.txt
```

## Run the analyzer

```bash
python run.py path/to/your_intro.mp4
```

### Important

When using a global environment:

Do **not** run:

```bash
uv sync
python -m venv .venv
```

These commands create a new project environment.

------

# Usage

**With OpenAI (cloud, needs API key):**

```bash
uv run python run.py path/to/your_intro.mp4
# or with pip: python run.py path/to/your_intro.mp4
```

**With Ollama (local, no API key or budget):**

```bash
python run.py path/to/your_intro.mp4 --llm-model llama3
# or: --llm-model mistral
```

**Custom instructions (Ollama):** use a Markdown or text file to define evaluation criteria; the tool builds a prompt, shows it for confirmation, and optionally lets you edit it. The prompt is saved for all future Ollama runs:

```bash
python run.py path/to/intro.mp4 --llm-model llama3 --instructions path/to/criteria.md
```

- The generated prompt is shown on the command line. You are asked: **Modify this prompt? (yes/y / no/n)**. If you choose yes, your editor (e.g. `EDITOR` or Notepad on Windows) opens so you can edit the prompt; the result is saved.
- Saved prompt path: `saved_ollama_prompt.txt` in the project directory. On later runs with `--llm-model`, this file is used automatically (no need to pass `--instructions` again).

**Batch (folder):** process all videos in a folder and write one JSON per video to an output directory:

```bash
python run.py --folder path/to/videos --output output
# Default --output is "output"; JSON files are named <videostem>.json
```

Optional: use a larger Whisper model for better accuracy (slower):

```bash
python run.py path/to/intro.mp4 --whisper-model small
```

Whisper models:

```
tiny
base
small
medium
large
large-v3
```

------

# Example output

**Single video (stdout):**

```
Transcript:
Hello, my name is Parminder Singh...

Feedback:
- Good introduction structure
- Slightly fast pacing
- Repeated filler words: "uh", "actually"
- Confidence score: 7.5/10
- Suggestion: pause more after key sentences
```

**Folder mode (in `--output`):** for each video you get:

- **`<stem>.json`** — full result (video, transcript, feedback).
- **`<stem>.txt`** — plain text (video name, transcript, feedback) for easy copy-paste.
- **`results.csv`** — one CSV with all videos: columns `video`, `transcript`, `feedback`, score columns, `suggestion`. Scores are parsed from the feedback; open in Excel or any spreadsheet.
- If the analysis cannot be parsed into the expected format (e.g. custom prompt produced different output), a **`<stem>_feedback_readable.txt`** file is also written with a clear, readable presentation of the transcript and feedback.

Example JSON:

```json
{
  "video": "nikhil-intro.mp4",
  "transcript": "Hello, my name is...",
  "feedback": "- Good introduction structure\n- Slightly fast pacing\n..."
}
```

------

# Later additions

- Speaking speed (words per minute)
- Pause detection
- Facial expression / posture (OpenCV)
- Eye contact estimation
- Gesture analysis
