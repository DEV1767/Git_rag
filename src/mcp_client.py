import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

client = MultiServerMCPClient(
    {
        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
                "stdio",
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN":
                    os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            },
            "transport": "stdio",
        }
    }
)


async def main():
    tools = await client.get_tools()

    print("Available GitHub tools:")

    for tool in tools:
        print("-", tool.name)


asyncio.run(main())