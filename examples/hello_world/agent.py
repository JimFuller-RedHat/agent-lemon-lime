"""Hello World agent — minimal pydantic-ai agent for testing agent-lemon-lime."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

agent = Agent(
    "anthropic:claude-haiku-4-5",
    system_prompt="You are a friendly assistant. When greeted, respond with 'Hello, World!'.",
)


async def main() -> None:
    result = await agent.run("Say hello.")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
