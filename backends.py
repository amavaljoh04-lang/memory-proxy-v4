"""Multi-backend LLM support — Ollama, OpenAI API, Grok, Anthropic, etc.

The proxy can forward to any OpenAI-compatible API endpoint.
Configuration via environment variables or /backend slash command.

Each backend has:
  - name: display name
  - base_url: API base URL
  - api_key: optional API key
  - models: list of available models (auto-discovered for Ollama)
  - type: "ollama" or "openai" (protocol)
"""
import os
import json
import logging
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("backends")


@dataclass
class Backend:
    name: str
    base_url: str
    api_key: str = ""
    backend_type: str = "ollama"  # "ollama" or "openai"
    models: list = field(default_factory=list)
    default_model: str = ""


# Built-in backends — users can add more via /backend add
_BACKENDS: dict[str, Backend] = {}
_active_backend: str = "ollama"


def init_backends():
    """Initialize backends from environment variables."""
    global _active_backend

    # Always register Ollama (local)
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    _BACKENDS["ollama"] = Backend(
        name="Ollama (local)",
        base_url=ollama_url,
        backend_type="ollama",
    )

    # OpenAI API (if key provided)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        _BACKENDS["openai"] = Backend(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=openai_key,
            backend_type="openai",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            default_model="gpt-4o-mini",
        )

    # Grok / xAI (if key provided)
    grok_key = os.getenv("GROK_API_KEY", os.getenv("XAI_API_KEY", ""))
    if grok_key:
        _BACKENDS["grok"] = Backend(
            name="Grok (xAI)",
            base_url="https://api.x.ai/v1",
            api_key=grok_key,
            backend_type="openai",
            models=["grok-2", "grok-2-mini"],
            default_model="grok-2-mini",
        )

    # Anthropic Claude (if key provided)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        _BACKENDS["anthropic"] = Backend(
            name="Anthropic Claude",
            base_url="https://api.anthropic.com/v1",
            api_key=anthropic_key,
            backend_type="anthropic",
            models=["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
            default_model="claude-sonnet-4-20250514",
        )

    # Mistral (if key provided)
    mistral_key = os.getenv("MISTRAL_API_KEY", "")
    if mistral_key:
        _BACKENDS["mistral"] = Backend(
            name="Mistral AI",
            base_url="https://api.mistral.ai/v1",
            api_key=mistral_key,
            backend_type="openai",
            models=["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
            default_model="mistral-large-latest",
        )

    # Custom backend from env
    custom_url = os.getenv("CUSTOM_LLM_URL", "")
    custom_key = os.getenv("CUSTOM_LLM_KEY", "")
    if custom_url:
        _BACKENDS["custom"] = Backend(
            name="Custom LLM",
            base_url=custom_url,
            api_key=custom_key,
            backend_type="openai",
        )

    log.info(f"Backends initialized: {', '.join(_BACKENDS.keys())}")


def get_active_backend() -> Backend:
    """Get the currently active backend."""
    return _BACKENDS.get(_active_backend, _BACKENDS.get("ollama"))


def get_active_backend_name() -> str:
    return _active_backend


def set_active_backend(name: str) -> bool:
    """Switch active backend. Returns True if successful."""
    global _active_backend
    if name.lower() in _BACKENDS:
        _active_backend = name.lower()
        return True
    return False


def add_backend(name: str, base_url: str, api_key: str = "",
                backend_type: str = "openai", models: list = None) -> Backend:
    """Add a new backend at runtime."""
    backend = Backend(
        name=name,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        backend_type=backend_type,
        models=models or [],
    )
    _BACKENDS[name.lower()] = backend
    return backend


def list_backends() -> dict[str, dict]:
    """List all available backends."""
    result = {}
    for name, b in _BACKENDS.items():
        result[name] = {
            "name": b.name,
            "type": b.backend_type,
            "base_url": b.base_url[:50] + "..." if len(b.base_url) > 50 else b.base_url,
            "has_key": bool(b.api_key),
            "models": b.models,
            "active": name == _active_backend,
        }
    return result


def resolve_model(model_name: str) -> tuple[Backend, str]:
    """Resolve a model name to (backend, actual_model_name).
    
    If model_name contains '/' like 'openai/gpt-4o', route to that backend.
    Otherwise use the active backend.
    """
    if "/" in model_name:
        parts = model_name.split("/", 1)
        backend_name = parts[0].lower()
        actual_model = parts[1]
        if backend_name in _BACKENDS:
            return _BACKENDS[backend_name], actual_model

    backend = get_active_backend()
    return backend, model_name


async def forward_chat_ollama(client: httpx.AsyncClient, backend: Backend,
                               body: dict, timeout: float = 300.0):
    """Forward a chat request to an Ollama backend."""
    url = f"{backend.base_url}/api/chat"
    return await client.stream("POST", url, json=body, timeout=timeout)


async def forward_chat_openai(client: httpx.AsyncClient, backend: Backend,
                                body: dict, timeout: float = 300.0):
    """Forward a chat request to an OpenAI-compatible backend.
    
    Converts Ollama format → OpenAI format.
    """
    # Convert body to OpenAI format
    openai_body = {
        "model": body.get("model", backend.default_model),
        "messages": body.get("messages", []),
        "stream": body.get("stream", True),
    }
    if "options" in body:
        if "temperature" in body["options"]:
            openai_body["temperature"] = body["options"]["temperature"]
        if "num_predict" in body["options"]:
            openai_body["max_tokens"] = body["options"]["num_predict"]

    headers = {"Content-Type": "application/json"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"

    url = f"{backend.base_url}/chat/completions"
    return await client.stream("POST", url, json=openai_body,
                                headers=headers, timeout=timeout)


def parse_ollama_stream_line(line: str) -> tuple[str, bool]:
    """Parse an Ollama streaming line. Returns (content, is_done)."""
    try:
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        done = data.get("done", False)
        return content, done
    except json.JSONDecodeError:
        return "", False


def parse_openai_stream_line(line: str) -> tuple[str, bool]:
    """Parse an OpenAI SSE streaming line. Returns (content, is_done)."""
    if not line.startswith("data: "):
        return "", False
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return "", True
    try:
        data = json.loads(data_str)
        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
        finish = data.get("choices", [{}])[0].get("finish_reason")
        return content, finish == "stop"
    except (json.JSONDecodeError, IndexError):
        return "", False
