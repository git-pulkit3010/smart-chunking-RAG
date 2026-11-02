"""
Unit tests for the NVIDIA-backed generator and configuration wiring.
"""

import pytest

from src.config import DEFAULT_NVIDIA_BASE_URL, Settings
from src.generator import _iter_sse_content, build_rag_prompt, stream_answer


def test_settings_reads_nvidia_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

    settings = Settings.load()
    assert settings.nvidia_api_key == "nvapi-test-key"
    assert settings.nvidia_model == "meta/llama-3.3-70b-instruct"
    assert settings.nvidia_base_url == DEFAULT_NVIDIA_BASE_URL


def test_settings_nvidia_base_url_override(monkeypatch):
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.test/v1/")

    settings = Settings.load()
    assert settings.nvidia_base_url == "https://example.test/v1"


def test_settings_nvidia_defaults_empty(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)

    settings = Settings.load()
    assert settings.nvidia_api_key is None
    assert settings.nvidia_model == ""


def test_stream_answer_requires_api_key():
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is not set"):
        stream_answer("q", [], api_key="", model="meta/llama-3.3-70b-instruct")


def test_stream_answer_requires_model():
    with pytest.raises(ValueError, match="NVIDIA_MODEL is not set"):
        stream_answer("q", [], api_key="nvapi-test-key", model="")


def test_iter_sse_content_parses_deltas():
    lines = [
        "",
        ": comment",
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        'data: {"choices": [{"delta": {}}]}',
        "data: not-json",
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "ignored"}}]}',
    ]
    assert list(_iter_sse_content(iter(lines))) == ["Hello", " world"]


def test_build_rag_prompt_includes_context():
    hits = [
        {
            "text": "Chunk body text.",
            "meta": {"heading": "Introduction", "page": 2, "paper_title": "My Paper"},
        }
    ]
    prompt = build_rag_prompt("What is X?", hits)
    assert "What is X?" in prompt
    assert "Introduction" in prompt
    assert "My Paper" in prompt
    assert "Chunk body text." in prompt


def test_stream_answer_posts_to_nvidia(monkeypatch):
    """Verify the request payload/URL and that streamed deltas are assembled."""
    captured = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "Ans"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "wer"}}]}'
            yield "data: [DONE]"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, headers=None, json=None):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)

    tokens = []
    result = stream_answer(
        "Q?",
        [{"text": "ctx", "meta": {"heading": "S", "page": 1, "paper_title": "P"}}],
        api_key="nvapi-test-key",
        model="meta/llama-3.3-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        token_callback=tokens.append,
    )

    assert result == "Answer"
    assert tokens == ["Ans", "wer"]
    assert captured["method"] == "POST"
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer nvapi-test-key"
    assert captured["payload"]["model"] == "meta/llama-3.3-70b-instruct"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert "Q?" in captured["payload"]["messages"][1]["content"]


def test_stream_answer_raises_on_http_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{\"detail\": \"invalid api key\"}"

        def iter_lines(self):
            return iter([])

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, headers=None, json=None):
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="401"):
        stream_answer("Q?", [], api_key="bad", model="meta/llama-3.3-70b-instruct")
