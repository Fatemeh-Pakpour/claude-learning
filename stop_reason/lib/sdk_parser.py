"""Pretty-printing helpers for Claude Agent SDK messages.

Every message yielded by `query()` / `ClaudeSDKClient.receive_response()` is
passed through `log_message()`, which prints a readable, colour-coded summary
that highlights `stop_reason` and the tool_use / tool_result round trip.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"

STOP_REASON_COLORS = {
    "tool_use": YELLOW,
    "end_turn": GREEN,
    "max_tokens": RED,
    "stop_sequence": MAGENTA,
    "refusal": RED,
    "pause_turn": MAGENTA,
}


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def format_stop_reason(stop_reason: str | None) -> str:
    if stop_reason is None:
        return f"{DIM}None{RESET}"
    color = STOP_REASON_COLORS.get(stop_reason, CYAN)
    return f"{BOLD}{color}{stop_reason}{RESET}"


def _tool_result_text(content: Any) -> str:
    """Tool result content is either a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item["text"])
        else:
            parts.append(_json(item))
    return "\n".join(parts)


def format_block(block: Any) -> str:
    if isinstance(block, TextBlock):
        return f"{CYAN}[text]{RESET}\n{_indent(block.text)}"
    if isinstance(block, ThinkingBlock):
        return f"{DIM}[thinking]\n{_indent(block.thinking)}{RESET}"
    if isinstance(block, ToolUseBlock):
        return (
            f"{YELLOW}[tool_use]{RESET} name={block.name} id={block.id}\n"
            f"{_indent(_json(block.input))}"
        )
    if isinstance(block, ToolResultBlock):
        flag = f" {RED}is_error=True{RESET}" if block.is_error else ""
        return (
            f"{GREEN}[tool_result]{RESET} tool_use_id={block.tool_use_id}{flag}\n"
            f"{_indent(_tool_result_text(block.content))}"
        )
    return f"[{type(block).__name__}] {block!r}"


def _header(title: str, color: str) -> str:
    return f"\n{BOLD}{color}=== {title} {'=' * max(0, 60 - len(title))}{RESET}"


def stream_stop_reason(message: Any) -> str | None:
    """Return the API stop_reason if `message` is a raw `message_delta` event.

    Requires `include_partial_messages=True`. The CLI streams each content
    block as its own AssistantMessage, so `AssistantMessage.stop_reason` is
    usually None; the real per-turn value arrives in this event.
    """
    if not isinstance(message, StreamEvent) or message.parent_tool_use_id:
        return None
    event = message.event
    if event.get("type") != "message_delta":
        return None
    return event.get("delta", {}).get("stop_reason")


def log_message(message: Any, index: int | None = None) -> None:
    """Print one SDK message in a readable form."""
    tag = f"#{index} " if index is not None else ""

    if isinstance(message, StreamEvent):
        reason = stream_stop_reason(message)
        if reason is not None:
            print(
                f"\n  {BOLD}>>> API turn ended (message_delta): "
                f"stop_reason={format_stop_reason(reason)}"
            )
        return  # other token-level stream events are too noisy to print

    if isinstance(message, RateLimitEvent):
        return

    if isinstance(message, SystemMessage):
        print(_header(f"{tag}SystemMessage ({message.subtype})", DIM))
        data = message.data or {}
        if "model" in data:
            print(f"  model: {data['model']}")
        if "tools" in data:
            mcp_tools = [t for t in data["tools"] if t.startswith("mcp__")]
            print(f"  mcp tools: {mcp_tools}")
        return

    if isinstance(message, AssistantMessage):
        print(_header(f"{tag}AssistantMessage  (role=assistant)", CYAN))
        print(f"  stop_reason: {format_stop_reason(message.stop_reason)}")
        for block in message.content:
            if isinstance(block, ThinkingBlock) and not block.thinking.strip():
                print(f"  {DIM}[thinking] (redacted){RESET}")
                continue
            print(_indent(format_block(block), "  "))
        return

    if isinstance(message, UserMessage):
        print(_header(f"{tag}UserMessage  (role=user, tool results appended)", GREEN))
        content = message.content
        if isinstance(content, str):
            print(_indent(content, "  "))
        else:
            for block in content:
                print(_indent(format_block(block), "  "))
        return

    if isinstance(message, ResultMessage):
        print(_header(f"{tag}ResultMessage  (final)", MAGENTA))
        print(f"  subtype:     {message.subtype}")
        print(f"  stop_reason: {format_stop_reason(message.stop_reason)}")
        print(f"  is_error:    {message.is_error}")
        print(f"  num_turns:   {message.num_turns}")
        print(f"  duration_ms: {message.duration_ms}")
        if message.total_cost_usd is not None:
            print(f"  cost_usd:    {message.total_cost_usd:.6f}")
        if message.result:
            print(f"  result:\n{_indent(message.result)}")
        return

    print(_header(f"{tag}{type(message).__name__}", DIM))
    print(_indent(repr(message), "  "))


def log_conversation(history: list[dict[str, Any]]) -> None:
    """Print the reconstructed API-style conversation (role + content blocks)."""
    print(_header("Conversation history (as the model sees it)", BOLD))
    for i, turn in enumerate(history):
        role = turn["role"]
        color = GREEN if role == "user" else CYAN
        summary = []
        for block in turn["content"]:
            kind = block["type"]
            if kind == "text":
                summary.append(f"text: {block['text'][:70]!r}")
            elif kind == "tool_use":
                summary.append(f"tool_use: {block['name']}({json.dumps(block['input'])})")
            elif kind == "tool_result":
                summary.append(f"tool_result -> {block['tool_use_id']}")
        print(f"  [{i}] {color}{role:<9}{RESET} " + " | ".join(summary))
