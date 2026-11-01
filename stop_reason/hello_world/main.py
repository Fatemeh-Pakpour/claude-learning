import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions
async def main():
    async for message in query(
        prompt="Review hello.rb for bugs",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],  # Auto-approve these tools
            permission_mode="acceptEdits",  # Auto-approve file edits
        ),
    ):
        print(message)

asyncio.run(main())
