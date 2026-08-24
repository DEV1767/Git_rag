import asyncio
import json
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from llm import embedding, Groq_model, google_model

load_dotenv()


async def main():

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

            result = await session.list_tools()

            tools = result.tools

            print(f"\nAvailable GitHub tools: {len(tools)}\n")

            for mcp_tool in tools:
                print(mcp_tool.name)
                print(mcp_tool.description)
                print("-" * 50)

            vectorstore = await create_tool_index(tools)

            query = input("\nAsk something about GitHub: ")

            retrieved_docs = retriver_tools(vectorstore, query, k=5)

            decision = await choose_tool(query, retrieved_docs)

            print("\nAgent decision:")
            print(json.dumps(decision, indent=2))

            tool_name = decision["tool_name"]

            arguments = decision.get("arguments", {})

            available_tool_names = {tool.name for tool in tools}

            if tool_name not in available_tool_names:
                print(f"\nERROR: {tool_name} is not an available MCP tool.")
                return

            tool_result = await session.call_tool(tool_name, arguments)

            result_text = extract_mcp_result(tool_result)

            print("\nMCP Result:")
            print(result_text)

            final_answer = await generate_answer(query, result_text)

            print("\nFinal Answer:")
            print(final_answer)


async def create_tool_index(tools):

    if os.path.exists("tool_store"):

        vectorstore = FAISS.load_local(
            "tool_store", embedding, allow_dangerous_deserialization=True
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


def retriver_tools(vectorstore, query, k=5):

    documents = vectorstore.similarity_search(query, k=k)

    print("\nRelevant tools:")

    for doc in documents:
        print("-", doc.metadata["tool_name"])

    return documents


async def choose_tool(query, retrived_docs):

    tools_description = "\n\n".join(doc.page_content for doc in retrived_docs)

    prompt = f"""
You are an intelligent GitHub agent.

USER QUESTION:

{query}

AVAILABLE MCP TOOLS:

{tools_description}

Choose the best MCP tool for the user's request.

Return ONLY valid JSON:

{{
    "tool_name": "exact_tool_name",
    "arguments": {{}}
}}

The tool_name must exactly match an available MCP tool.

Do not invent tools.

Do not add explanations.
"""

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    content = content.replace("```json", "")

    content = content.replace("```", "")

    content = content.strip()

    return json.loads(content)


def extract_mcp_result(tool_result):

    output = []

    for content in tool_result.content:

        if hasattr(content, "text"):
            output.append(content.text)

        else:
            output.append(str(content))

    return "\n".join(output)


async def generate_answer(query, tool_result):

    prompt = f"""
You are a helpful GitHub assistant.

USER QUESTION:

{query}

RESULT FROM GITHUB:

{tool_result}

Answer the user's question using the GitHub result.

Do not invent information.
"""

    response = await google_model.ainvoke(prompt)

    return response.content


if __name__ == "__main__":
    asyncio.run(main())
