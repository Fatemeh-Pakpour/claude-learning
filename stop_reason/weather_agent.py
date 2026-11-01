"""stop_reason demo with the Claude Agent SDK and a weather tool.

Flow you will see in the output:

  1. user prompt           -> "What's the weather in Paris and Tokyo?"
  2. AssistantMessage      -> stop_reason = "tool_use"   (Claude asks for get_weather)
  3. UserMessage           -> tool_result blocks appended to the conversation
  4. AssistantMessage      -> stop_reason = "end_turn"   (Claude answers in text)
  5. ResultMessage         -> final stop_reason of the whole run

Run:
    pip install -r requirements.txt
    python weather_agent.py
    python weather_agent.py "Is it raining in London?"
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

from lib.sdk_parser import (
    format_stop_reason,
    log_conversation,
    log_message,
    stream_stop_reason,
)

# Fake data keeps the demo deterministic and free of API keys.
FAKE_WEATHER = {
    "paris": {"temp_c": 14, "condition": "light rain", "humidity": 82},
    "tokyo": {"temp_c": 19, "condition": "clear", "humidity": 55},
    "london": {"temp_c": 11, "condition": "overcast", "humidity": 77},
    "new york": {"temp_c": 9, "condition": "windy", "humidity": 48},
}


@tool(
    "get_weather",
    "Get the current weather for a city. Call once per city.",
    {"city": str},
)
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    city = args["city"]
    data = FAKE_WEATHER.get(city.strip().lower())
    if data is None:
        return {
            "content": [{"type": "text", "text": f"No weather data for '{city}'."}],
            "is_error": True,
        }
    text = (
        f"{city}: {data['temp_c']}°C, {data['condition']}, "
        f"humidity {data['humidity']}%"
    )
    return {"content": [{"type": "text", "text": text}]}


SERVER_NAME = "weather"
TOOL_NAME = f"mcp__{SERVER_NAME}__get_weather"


def record(history: list[dict[str, Any]], message: Any) -> None:
    """Mirror the SDK messages into an API-style message list.

    The CLI may stream one assistant turn as several AssistantMessages (one per
    content block), so consecutive blocks from the same role are merged.
    """
    if isinstance(message, AssistantMessage):
        role, blocks = "assistant", []
        for b in message.content:
            if isinstance(b, TextBlock):
                blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
    elif isinstance(message, UserMessage) and not isinstance(message.content, str):
        role = "user"
        blocks = [
            {"type": "tool_result", "tool_use_id": b.tool_use_id, "content": b.content}
            for b in message.content
            if isinstance(b, ToolResultBlock)
        ]
    else:
        return

    if not blocks:
        return
    if history and history[-1]["role"] == role:
        history[-1]["content"].extend(blocks)
    else:
        history.append({"role": role, "content": blocks})


async def main(prompt: str) -> None:
    server = create_sdk_mcp_server(SERVER_NAME, tools=[get_weather])
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a weather assistant. Always use the get_weather tool "
            "to look up weather; never guess."
        ),
        mcp_servers={SERVER_NAME: server},
        allowed_tools=[TOOL_NAME],
        # Isolate the demo: no built-in tools, no user MCP servers or settings.
        tools=[],
        strict_mcp_config=True,
        setting_sources=[],
        # Needed to receive raw `message_delta` events carrying stop_reason.
        include_partial_messages=True,
        max_turns=5,
    )

    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    stop_reasons: list[str | None] = []
    final: str | None = None

    print(f"User prompt: {prompt}")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for i, message in _enumerate(client.receive_response()):
            log_message(message, i)
            record(history, message)

            reason = stream_stop_reason(message)
            if reason is not None:
                stop_reasons.append(reason)
            elif isinstance(message, UserMessage):
                log_conversation(history)
            elif isinstance(message, ResultMessage):
                final = message.stop_reason

    log_conversation(history)

    print("\nstop_reason timeline (per API turn, from message_delta):")
    for n, reason in enumerate(stop_reasons, 1):
        print(f"  turn {n}: {format_stop_reason(reason)}")
    print(f"  final (ResultMessage): {format_stop_reason(final)}")


async def _enumerate(aiter):
    i = 0
    async for item in aiter:
        yield i, item
        i += 1


if __name__ == "__main__":
    user_prompt = (
        " ".join(sys.argv[1:]) or "What's the weather like in Paris and Tokyo right now?"
    )
    asyncio.run(main(user_prompt))
