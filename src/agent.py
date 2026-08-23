import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tool_router import select_tools

from mcp_client import client

load_dotenv()


async def main():

    all_tools = await client.get_tools()
    tools = select_tools(all_tools, "issues")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    agent = create_agent(model=llm, tools=tools)

    response = await agent.ainvoke(
        {"messages": [("user", "What is my GitHub username?")]}
    )

    print(response["messages"][-1].content)


asyncio.run(main())
