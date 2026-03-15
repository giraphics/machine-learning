"""Analyze transcript with an LLM for speaking/confidence feedback."""

import os
from openai import OpenAI

# Ollama exposes an OpenAI-compatible API at localhost:11434
OLLAMA_BASE_URL = "http://localhost:11434/v1"

EVALUATION_PROMPT = """You are an expert coach for self-introductions and public speaking. Analyze the following transcript of a self-introduction and give structured feedback.

Transcript:
---
{transcript}
---

Provide your analysis in this exact format. Be concise and actionable.

Feedback:
- [2-4 bullet points: structure, pacing, clarity, standout positives]
- Repeated filler words: [list any: "um", "uh", "like", "actually", etc., or "none" if minimal]
- Clarity score: X/10
- Fluency score: X/10
- Confidence score: X/10
- Structure score: X/10
- Vocabulary score: X/10
- Suggestion: [one concrete, specific tip—e.g., "Pause more after key sentences" or "Slow down in the first 30 seconds"]
"""


def analyze_transcript(
    transcript: str,
    api_key: str | None = None,
    llm_model: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """
    Send transcript to an LLM for evaluation. Returns the model's feedback text.

    If llm_model is set (e.g. "llama3"), use Ollama locally. Otherwise use OpenAI
    (requires OPENAI_API_KEY). If prompt_template is provided it must contain {transcript}.
    """
    if prompt_template and "{transcript}" in prompt_template:
        prompt = prompt_template.format(transcript=transcript)
    else:
        prompt = EVALUATION_PROMPT.format(transcript=transcript)
    messages = [
        {"role": "system", "content": "You are a concise, constructive speaking coach. Reply only with the requested feedback format."},
        {"role": "user", "content": prompt},
    ]

    if llm_model:
        # Local Ollama (no API key needed)
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        model = llm_model
    else:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Use .env or pass api_key=..., or run locally with --llm-model llama3"
            )
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        if llm_model:
            msg = str(e).lower()
            if "connection" in msg or "refused" in msg or "connect" in msg:
                raise RuntimeError(
                    f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
                    "Start Ollama (e.g. run 'ollama serve' or open the Ollama app) and ensure the model is pulled: ollama pull " + llm_model
                ) from e
        else:
            msg = str(e).lower()
            if getattr(e, "status_code", None) == 429 or "429" in msg or "quota" in msg or "insufficient_quota" in msg or type(e).__name__ == "RateLimitError":
                raise RuntimeError(
                    "OpenAI API quota exceeded (429). Check your plan and billing at "
                    "https://platform.openai.com/account/billing — or use a different API key."
                ) from e
        raise
