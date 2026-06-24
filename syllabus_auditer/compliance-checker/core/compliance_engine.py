"""
Sends each compliance rule + document text to the selected AI provider and returns structured results.

Supported providers (all use the OpenAI-compatible SDK):
  - MiniMax M3   (https://api.minimax.chat/v1)
  - Groq         (https://api.groq.com/openai/v1)
  - Google Gemini (https://generativelanguage.googleapis.com/v1beta/openai/)
  - Ollama       (http://localhost:11434/v1  — runs locally, free)
"""

import json
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI


# ── Provider registry ─────────────────────────────────────────────────────────

PROVIDERS = {
    "xAI Grok": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3",
        "needs_key": True,
        "cost": "Pay per token — see x.ai/api",
        "help": "Get key at https://console.x.ai/",
    },
    "MiniMax M3": {
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-M3",
        "needs_key": True,
        "cost": "~$0.03–0.08 per audit",
        "help": "Get key at https://www.minimaxi.com/",
    },
    "Groq (Free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "cost": "Free tier: 14,400 req/day",
        "help": "Get key at https://console.groq.com/",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-1.5-flash",
        "needs_key": True,
        "cost": "Free tier: 1,500 req/day",
        "help": "Get key at https://aistudio.google.com/apikey",
    },
    "Ollama (Local — Free)": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:14b",
        "needs_key": False,
        "cost": "Free — runs on your computer",
        "help": "Install Ollama at https://ollama.com then run: ollama pull qwen2.5:14b",
    },
}

OLLAMA_SUGGESTED_MODELS = [
    "qwen2.5:14b",
    "llama3.1",
    "llama3.1:70b",
    "mistral",
    "gemma2:9b",
]


def get_ollama_models() -> list[str]:
    """
    Queries the local Ollama server for installed models.
    Returns a list of model names, or the static fallback list if Ollama is not running.
    """
    try:
        import urllib.request
        import json as _json
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = _json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return models if models else OLLAMA_SUGGESTED_MODELS
    except Exception:
        return OLLAMA_SUGGESTED_MODELS


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    rule: str
    result: Literal["pass", "fail", "uncertain"]
    confidence: float      # 0.0 to 1.0
    evidence: str          # quote from the document
    reason: str            # one-sentence explanation


@dataclass
class ProviderConfig:
    provider_name: str
    base_url: str
    model: str
    api_key: str


# ── Public API ────────────────────────────────────────────────────────────────

def check_document(document_text: str, rules: list[str], config: ProviderConfig) -> list[RuleResult]:
    """Evaluates a document against every rule. Returns one RuleResult per rule."""
    client = OpenAI(
        api_key=config.api_key or "ollama",   # Ollama ignores the key but SDK requires a value
        base_url=config.base_url,
    )
    return [_evaluate_single_rule(client, document_text, rule, config.model) for rule in rules]


# ── Internal ──────────────────────────────────────────────────────────────────

def _evaluate_single_rule(client: OpenAI, document_text: str, rule: str, model: str) -> RuleResult:
    truncated = document_text[:12000]
    if len(document_text) > 12000:
        truncated += "\n\n[... document truncated for length ...]"

    prompt = f"""You are a strict compliance checker for academic documents.

Evaluate whether the following document satisfies this compliance requirement:

REQUIREMENT:
{rule}

DOCUMENT:
{truncated}

Respond ONLY with valid JSON in exactly this format:
{{
  "result": "pass" or "fail" or "uncertain",
  "confidence": <number between 0.0 and 1.0>,
  "evidence": "<exact quote from the document that supports your decision, or 'Not found' if absent>",
  "reason": "<one sentence explaining your decision>"
}}

Be strict. If the requirement is partially met but not fully, respond with "uncertain"."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return RuleResult(
            rule=rule,
            result=data.get("result", "uncertain"),
            confidence=float(data.get("confidence", 0.5)),
            evidence=data.get("evidence", ""),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        return RuleResult(
            rule=rule,
            result="uncertain",
            confidence=0.0,
            evidence="",
            reason=f"Error during evaluation: {e}",
        )
