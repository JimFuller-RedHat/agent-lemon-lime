"""Shared LLM call helpers for Anthropic and OpenAI."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def _call_anthropic(
    model_name: str, system_prompt: str, user_content: str, max_tokens: int
) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM call")
        return None
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model_name,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.warning("Anthropic API returned %d — skipping LLM call", resp.status_code)
            return None
        data = resp.json()
        text = data["content"][0]["text"]
        return text if text.strip() else None
    except Exception:
        logger.warning("Anthropic API call failed — skipping LLM call", exc_info=True)
        return None


def _call_openai(
    model_name: str, system_prompt: str, user_content: str, max_tokens: int
) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM call")
        return None
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.warning("OpenAI API returned %d — skipping LLM call", resp.status_code)
            return None
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text if text.strip() else None
    except Exception:
        logger.warning("OpenAI API call failed — skipping LLM call", exc_info=True)
        return None


def call_llm(
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4096,
) -> str | None:
    """Call an LLM via its HTTP API.

    Args:
        model: Provider and model name in format 'provider/model_name'
            (e.g., 'anthropic/claude-sonnet-4-20250514' or 'openai/gpt-4o')
        system_prompt: System prompt for the LLM
        user_content: User message content
        max_tokens: Maximum tokens in response (default: 4096)

    Returns:
        The LLM's response text, or None if the call failed
    """
    parts = model.split("/", 1)
    if len(parts) != 2:
        logger.warning("Invalid model format '%s' — expected 'provider/model'", model)
        return None
    provider, model_name = parts

    if provider == "anthropic":
        return _call_anthropic(model_name, system_prompt, user_content, max_tokens)
    if provider == "openai":
        return _call_openai(model_name, system_prompt, user_content, max_tokens)

    logger.warning("Unknown provider '%s' — skipping LLM call", provider)
    return None
