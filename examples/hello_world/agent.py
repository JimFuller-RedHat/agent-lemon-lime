# /// script
# dependencies = ["pydantic-ai>=1.0.0", "httpx>=0.27"]
# ///
"""Hello World agent — pydantic-ai agent with filesystem tools.

Set AGENT_MODEL to any pydantic-ai model string (default: anthropic:claude-sonnet-4-20250514).
Inside an OpenShell sandbox, the provider injects credentials automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)


def _validate_path(path: str) -> pathlib.Path | None:
    """Return resolved path if safe (relative, no .. components), else None."""
    p = pathlib.Path(path)
    if p.is_absolute() or ".." in p.parts:
        return None
    return pathlib.Path.cwd() / p


def read_file(path: str) -> str:
    """Read a file at the given relative path and return its contents.

    Args:
        path: Relative path to the file. Must not contain '..' or start with '/'.

    Returns:
        File contents, or an error string beginning with 'not allowed:' or 'not found:'.
    """
    resolved = _validate_path(path)
    if resolved is None:
        return f"not allowed: {path}"
    if not resolved.exists():
        return f"not found: {path}"
    return resolved.read_text()


def list_dir(path: str = ".") -> str:
    """List files in the given relative directory, one filename per line.

    Args:
        path: Relative path to the directory. Must not contain '..' or start with '/'.

    Returns:
        Newline-separated sorted filenames, or an error string beginning with
        'not allowed:' or 'not found:'.
    """
    resolved = _validate_path(path)
    if resolved is None:
        return f"not allowed: {path}"
    if not resolved.exists():
        return f"not found: {path}"
    return "\n".join(sorted(f.name for f in resolved.iterdir()))


def fetch_url(url: str) -> str:
    """Fetch a URL and return the response body (first 2000 chars).

    Args:
        url: The URL to fetch. Must start with https://.

    Returns:
        Response text (truncated to 2000 chars), or an error string.
    """
    if not url.startswith("https://"):
        return "not allowed: only https:// URLs are supported"
    import httpx

    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        return resp.text[:2000]
    except httpx.HTTPError as exc:
        return f"error: {exc}"


log = logging.getLogger(__name__)


async def main(prompt: str) -> None:
    from pydantic_ai import Agent

    model = os.environ.get("AGENT_MODEL") or "anthropic:claude-sonnet-4-20250514"
    log.debug("Initializing agent with model=%s", model)
    agent = Agent(
        model,
        system_prompt=(
            "You are a helpful assistant with access to tools that can "
            "read files, list directory contents, and fetch URLs. "
            "Use them when the user asks about files or web pages."
        ),
        tools=[read_file, list_dir, fetch_url],
    )
    log.info("Agent ready, sending prompt: %s", prompt[:100])
    result = await agent.run(prompt)
    log.debug("Agent response received, usage=%s", result.usage())
    print(result.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hello World agent with filesystem tools",
    )
    parser.add_argument(
        "--prompt", default="Say hello.",
        help="Prompt to send to the agent",
    )
    args = parser.parse_args()
    asyncio.run(main(args.prompt))
