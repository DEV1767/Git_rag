import asyncio
from mcp_setup import mcp_session


async def main():

    async with mcp_session() as session:

        result = await session.list_tools()

        for tool in result.tools:

            if tool.name == "issue_read":

                print("\nTOOL:")
                print(tool.name)

                print("\nDESCRIPTION:")
                print(tool.description)

                print("\nINPUT SCHEMA:")
                print(tool.inputSchema)


if __name__ == "__main__":
    asyncio.run(main())