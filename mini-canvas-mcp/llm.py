"""
llm.py - one small, reusable way to talk to different LLMs.

    Ollama  (local, free, DEFAULT)  http://localhost:11434
    Claude  (Anthropic API)         needs ANTHROPIC_API_KEY
    OpenAI  (Responses API)         needs OPENAI_API_KEY

Every provider has a different JSON format for messages and tool calls.
This file hides those differences behind ONE interface:

    chat = create_chat()                     # Ollama, auto-picks a model
    chat = create_chat("claude")             # or "openai", "ollama"
    chat.add_user("Draw a red circle")
    reply = chat.send(tools)                 # -> Reply(text, tool_calls)
    for call in reply.tool_calls:
        chat.add_tool_result(call, "Circle drawn.")
    reply = chat.send(tools)                 # model continues...

    print(chat.ask("Hello!"))                # quick text-only question

Tools are plain dicts:  {"name": ..., "description": ..., "parameters": <JSON schema>}
(MCP tools convert to this directly: name, description, input_schema.)

Nothing here knows about MCP or the Painter. Reuse it in any project.

Settings (all optional, via environment variables):
    LLM_PROVIDER        ollama | claude | openai     (default: ollama)
    LLM_MODEL           model name                   (default: see below)
    OLLAMA_HOST         default http://localhost:11434
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
    OPENAI_API_KEY,    OPENAI_BASE_URL

Try it:
    python llm.py                            # list local Ollama models + say hello
    python llm.py --provider claude "What is MCP in one sentence?"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx  # pip install httpx (see requirements.txt)

TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # local models can be slow

DEFAULT_MODELS = {
    "claude": "claude-sonnet-5-5",
    "openai": "gpt-5.5",
    # "ollama" is chosen automatically from the models installed on this PC
}

# Ollama models that are good at tool calling, best first.
# The first one that is installed on this PC is used.
OLLAMA_PREFERENCE = [
    "qwen3", "gpt-oss", "llama3.1", "qwen2.5", "llama3.3", "llama3.2",
    "mistral-small", "mistral", "granite3", "command-r", "phi4-mini",
]


# ============================================================================
# Common data types - the same for every provider
# ============================================================================

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


class Chat:
    """Base class. Each provider keeps the conversation in its OWN format."""

    provider = "base"

    def __init__(self, model: str, system: str | None = None):
        self.model = model
        self.system = system
        self.messages: list[dict] = []

    # --- the interface every provider implements ---
    def add_user(self, text: str) -> None:
        raise NotImplementedError

    def send(self, tools: list[dict] | None = None) -> Reply:
        """Send the conversation, store the model's answer, and return it."""
        raise NotImplementedError

    def add_tool_result(self, call: ToolCall, result: str) -> None:
        raise NotImplementedError

    # --- convenience ---
    def ask(self, text: str) -> str:
        """One question, one text answer (no tools)."""
        self.add_user(text)
        return self.send().text

    def __repr__(self) -> str:
        return f"<{self.provider}: {self.model}>"


def _post(url: str, payload: dict, headers: dict | None = None) -> dict:
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot connect to {url}. Is the service running?") from e
    except httpx.TimeoutException as e:
        raise LLMError(f"{url} took too long to answer. Try a smaller model.") from e
    if r.status_code >= 400:
        raise LLMError(f"{url} returned {r.status_code}: {r.text[:500]}")
    return r.json()


def _parse_args(raw) -> dict:
    """Arguments may arrive as a dict or as a JSON string, depending on provider."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


# ============================================================================
# Ollama - local models   (POST /api/chat)
# ============================================================================

class OllamaChat(Chat):
    provider = "ollama"

    def __init__(self, model: str | None = None, system: str | None = None):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        super().__init__(model or pick_ollama_model(self.host), system)
        if system:
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def send(self, tools=None):
        payload = {"model": self.model, "messages": self.messages, "stream": False}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }} for t in tools
            ]
        msg = _post(f"{self.host}/api/chat", payload)["message"]
        self.messages.append(msg)
        calls = [
            ToolCall(id=f"call_{i}", name=c["function"]["name"],
                     arguments=_parse_args(c["function"].get("arguments")))
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        return Reply(text=msg.get("content", ""), tool_calls=calls)

    def add_tool_result(self, call, result):
        self.messages.append({"role": "tool", "tool_name": call.name, "content": result})


def list_ollama_models(host: str | None = None) -> list[dict]:
    """Models installed on this PC, each with name, size and whether it can use tools."""
    host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=10).json().get("models", [])
    except httpx.HTTPError as e:
        raise LLMError(
            f"Cannot reach Ollama at {host}. Install it from https://ollama.com "
            "and make sure it is running."
        ) from e
    models = []
    for m in tags:
        try:
            info = httpx.post(f"{host}/api/show", json={"model": m["name"]}, timeout=10).json()
            caps = info.get("capabilities", [])
        except httpx.HTTPError:
            caps = []
        models.append({
            "name": m["name"],
            "size_gb": round(m.get("size", 0) / 1e9, 1),
            "tools": "tools" in caps,
            "embedding_only": caps == ["embedding"],
        })
    return models


def pick_ollama_model(host: str | None = None) -> str:
    """Choose the best installed model for tool calling."""
    models = [m for m in list_ollama_models(host) if not m["embedding_only"]]
    if not models:
        raise LLMError("No Ollama models installed. Try:  ollama pull qwen3:8b")
    with_tools = [m for m in models if m["tools"]] or models
    for family in OLLAMA_PREFERENCE:
        matches = [m for m in with_tools if m["name"].split(":")[0] == family]
        if matches:
            # several sizes of the same family -> take the smallest (fastest)
            return min(matches, key=lambda m: m["size_gb"])["name"]
    # nothing from the list: smallest model that supports tools
    return min(with_tools, key=lambda m: m["size_gb"])["name"]


# ============================================================================
# Claude - Anthropic Messages API   (POST /v1/messages)
# ============================================================================

class ClaudeChat(Chat):
    provider = "claude"

    def __init__(self, model: str | None = None, system: str | None = None):
        super().__init__(model or DEFAULT_MODELS["claude"], system)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("Set the ANTHROPIC_API_KEY environment variable to use Claude.")
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def send(self, tools=None):
        payload = {"model": self.model, "max_tokens": 4096, "messages": self.messages}
        if self.system:
            payload["system"] = self.system
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""),
                 "input_schema": t.get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = _post(f"{self.base}/v1/messages", payload, headers)
        blocks = data["content"]
        self.messages.append({"role": "assistant", "content": blocks})
        text = "".join(b["text"] for b in blocks if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {})
                 for b in blocks if b["type"] == "tool_use"]
        return Reply(text=text, tool_calls=calls)

    def add_tool_result(self, call, result):
        block = {"type": "tool_result", "tool_use_id": call.id, "content": result}
        # Claude wants all results for one turn inside ONE user message
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})


# ============================================================================
# OpenAI - Responses API   (POST /v1/responses)
# ============================================================================

class OpenAIChat(Chat):
    provider = "openai"

    def __init__(self, model: str | None = None, system: str | None = None):
        super().__init__(model or DEFAULT_MODELS["openai"], system)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("Set the OPENAI_API_KEY environment variable to use OpenAI.")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def send(self, tools=None):
        payload = {"model": self.model, "input": self.messages, "store": False}
        if self.system:
            payload["instructions"] = self.system
        if tools:
            payload["tools"] = [
                {"type": "function", "name": t["name"], "description": t.get("description", ""),
                 "parameters": t.get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        output = _post(f"{self.base}/responses", payload, headers)["output"]
        self.messages.extend(output)  # keep every output item for the next turn
        text, calls = "", []
        for item in output:
            if item["type"] == "message":
                text += "".join(c.get("text", "") for c in item["content"]
                                if c["type"] == "output_text")
            elif item["type"] == "function_call":
                calls.append(ToolCall(item["call_id"], item["name"],
                                      _parse_args(item.get("arguments"))))
        return Reply(text=text, tool_calls=calls)

    def add_tool_result(self, call, result):
        self.messages.append(
            {"type": "function_call_output", "call_id": call.id, "output": result})


# ============================================================================
# Factory - the one function most code needs
# ============================================================================

PROVIDERS = {"ollama": OllamaChat, "claude": ClaudeChat, "openai": OpenAIChat}


def create_chat(provider: str | None = None, model: str | None = None,
                system: str | None = None) -> Chat:
    """Create a chat. Defaults come from LLM_PROVIDER / LLM_MODEL, else Ollama."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    if provider in ("anthropic",):
        provider = "claude"
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider {provider!r}. Choose from {list(PROVIDERS)}.")
    model = model or os.environ.get("LLM_MODEL") or None
    return PROVIDERS[provider](model=model, system=system)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quick test of llm.py")
    ap.add_argument("question", nargs="?", default="Say hello to my students in one sentence.")
    ap.add_argument("--provider", choices=list(PROVIDERS))
    ap.add_argument("--model")
    args = ap.parse_args()

    if (args.provider or os.environ.get("LLM_PROVIDER", "ollama")) == "ollama":
        print("Ollama models on this PC:")
        for m in list_ollama_models():
            print(f"  {m['name']:<30} {m['size_gb']:>5} GB   tools: {'yes' if m['tools'] else 'no'}")

    chat = create_chat(args.provider, args.model)
    print(f"\nUsing {chat}\nQ: {args.question}\nA: {chat.ask(args.question)}")
