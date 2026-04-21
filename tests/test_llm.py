"""Tests for LLM call helpers."""

from unittest.mock import patch

import httpx

from agent_lemon_lime.report.llm import call_llm


def test_call_llm_anthropic_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "Hello from Claude"}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = call_llm(
            model="anthropic/claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
        )
    assert result == "Hello from Claude"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["x-api-key"] == "sk-test-key"
    assert call_kwargs.kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert call_kwargs.kwargs["json"]["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs.kwargs["json"]["max_tokens"] == 4096
    assert call_kwargs.kwargs["json"]["system"] == "You are a helpful assistant"
    assert call_kwargs.kwargs["json"]["messages"][0]["content"] == "Hello"
    assert call_kwargs.kwargs["timeout"] == 120.0


def test_call_llm_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Hello from GPT"}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = call_llm(
            model="openai/gpt-4o",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
        )
    assert result == "Hello from GPT"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "Bearer sk-test-key" in call_kwargs.kwargs["headers"]["Authorization"]
    assert call_kwargs.kwargs["json"]["model"] == "gpt-4o"
    assert call_kwargs.kwargs["json"]["max_tokens"] == 4096
    assert call_kwargs.kwargs["json"]["messages"][0]["role"] == "system"
    assert call_kwargs.kwargs["json"]["messages"][0]["content"] == "You are a helpful assistant"
    assert call_kwargs.kwargs["json"]["messages"][1]["role"] == "user"
    assert call_kwargs.kwargs["json"]["messages"][1]["content"] == "Hello"
    assert call_kwargs.kwargs["timeout"] == 120.0


def test_call_llm_anthropic_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = call_llm(
        model="anthropic/claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant",
        user_content="Hello",
    )
    assert result is None


def test_call_llm_openai_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = call_llm(
        model="openai/gpt-4o",
        system_prompt="You are a helpful assistant",
        user_content="Hello",
    )
    assert result is None


def test_call_llm_unknown_provider():
    result = call_llm(
        model="ollama/llama3",
        system_prompt="You are a helpful assistant",
        user_content="Hello",
    )
    assert result is None


def test_call_llm_invalid_model_format():
    result = call_llm(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant",
        user_content="Hello",
    )
    assert result is None


def test_call_llm_anthropic_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        500,
        json={"error": {"message": "Internal Server Error"}},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = call_llm(
            model="anthropic/claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
        )
    assert result is None


def test_call_llm_openai_api_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        500,
        json={"error": {"message": "Server Error"}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = call_llm(
            model="openai/gpt-4o",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
        )
    assert result is None


def test_call_llm_empty_response_returns_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": ""}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = call_llm(
            model="anthropic/claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
        )
    assert result is None


def test_call_llm_custom_max_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "Response"}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        call_llm(
            model="anthropic/claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant",
            user_content="Hello",
            max_tokens=2000,
        )
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["json"]["max_tokens"] == 2000
