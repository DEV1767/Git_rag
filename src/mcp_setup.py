import asyncio
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from llm import embedding, Groq_model

load_dotenv()


@asynccontextmanager
async def mcp_session():

    server = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
    )

    async with stdio_client(server) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            yield session


def extract_mcp_result(tool_result):

    output = []

    for content in tool_result.content:

        if hasattr(content, "text"):
            output.append(content.text)

        else:
            output.append(str(content))

    return "\n".join(output)


async def create_tool_index(tools):

    if os.path.exists("tool_store"):

        vectorstore = FAISS.load_local(
            "tool_store",
            embedding,
            allow_dangerous_deserialization=True,
        )

        print("Loaded existing MCP tool index")

        return vectorstore

    documents = []

    for mcp_tool in tools:

        tool_text = f"""
Tool Name:

{mcp_tool.name}

Description:

{mcp_tool.description}

Input Schema:

{json.dumps(mcp_tool.inputSchema, indent=2)}
"""

        document = Document(
            page_content=tool_text, metadata={"tool_name": mcp_tool.name}
        )

        documents.append(document)

    vectorstore = FAISS.from_documents(documents, embedding)

    vectorstore.save_local("tool_store")

    print(f"Indexed {len(documents)} MCP tools")

    return vectorstore


def retrieve_tools(vectorstore, query, k=5):

    documents = vectorstore.similarity_search(query, k=k)

    print("\nRelevant MCP tools:")

    for doc in documents:

        print("-", doc.metadata["tool_name"])

    return documents


async def choose_tool(query, retrieved_docs):

    tools_description = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are an intelligent GitHub agent.

USER QUESTION:

{query}


AVAILABLE MCP TOOLS:

{tools_description}


Choose the best MCP tool for the user's request.

Rules:

1. Select only one tool.
2. Use the exact tool name.
3. Generate arguments according to the Input Schema.
4. Do not invent tools.
5. Do not invent arguments.
6. Extract repository owner, repository name,
   file paths, search queries, etc. from the user question.
7. Return ONLY valid JSON.

Format:

{{
    "tool_name": "exact_tool_name",
    "arguments": {{
        "argument": "value"
    }}
}}
"""

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):

        content = "".join(str(x) for x in content)

    content = content.replace("```json", "")

    content = content.replace("```", "")

    content = content.strip()

    return json.loads(content)


async def search_with_mcp(session, query):

    result = await session.list_tools()

    tools = result.tools

    print(f"\nAvailable GitHub tools: {len(tools)}")

    vectorstore = await create_tool_index(tools)

    retrieved_docs = retrieve_tools(vectorstore, query, k=5)

    decision = await choose_tool(query, retrieved_docs)

    print("\nAgent decision:")

    print(json.dumps(decision, indent=2))

    tool_name = decision["tool_name"]

    arguments = decision.get("arguments", {})

    available_tool_names = {tool.name for tool in tools}

    if tool_name not in available_tool_names:

        raise ValueError(f"{tool_name} is not an available MCP tool.")

    print(f"\nCalling MCP tool: {tool_name}")

    tool_result = await session.call_tool(tool_name, arguments)

    result_text = extract_mcp_result(tool_result)

    return (tool_name, arguments, result_text)


async def search_repository(session, owner, repo):

    query = f"{repo} user:{owner}"

    result = await session.call_tool("search_repositories", {"query": query})

    result_text = extract_mcp_result(result)

    return result_text


async def main():

    async with mcp_session() as session:

        print("MCP connection successful.")

        result = await session.list_tools()

        print(f"Available tools: {len(result.tools)}")


if __name__ == "__main__":

    asyncio.run(main())
