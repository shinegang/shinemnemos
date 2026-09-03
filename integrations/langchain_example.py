"""LangChain example: ShineMnemos through the stdio bridge (langchain-mcp-adapters)."""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    async with MultiServerMCPClient({
        "shinemnemos": {
            "transport": "stdio",
            "command": "python3.12",
            "args": ["/ABSOLUTE/PATH/TO/shinemnemos/bridge/mnemos_bridge.py"],
            "env": {"MNEMOS_URL": "http://127.0.0.1:8765/"},
        }
    }) as client:
        tools = client.get_tools()
        print(tools)


if __name__ == "__main__":
    asyncio.run(main())
