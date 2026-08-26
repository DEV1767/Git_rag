import asyncio

from mcp_setup import (
    mcp_session,
    execute_mcp_tool,
)

from repo_search import build_repo_store

from retriver import (
    retrieve_documents,
    build_context,
)

from llm import Groq_model

from repo_helper import (
    parse_github_repo,
)

from qdrant_setup import build_tool_store

from prompt_helper import retriever_prompt

from tools.tool_selector import select_tool


async def generate_answer(
    question,
    context,
):
    prompt = retriever_prompt.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    return content


async def main():

    repository_url = input("Enter the repository url : \n")

    owner, repo = parse_github_repo(repository_url)

    tool_store = build_tool_store()

    async with mcp_session() as session:

        print("\nBuilding repository RAG store...")

        repo_store = await build_repo_store(
            session,
            owner,
            repo,
        )

        while True:

            question = input("\nEnter your Query (or type 'exit'):\n")

            if question.lower().strip() == "exit":
                break

            tool = await select_tool(
                question,
                session,
                tool_store,
            )

            selected_tool = tool["selection"]

            print("\nSelected tool:")
            print(selected_tool)

            arguments = {
                "owner": owner,
                "repo": repo,
            }

            mcp_result = await execute_mcp_tool(
                session,
                selected_tool,
                arguments,
            )

            print("\nMCP Result:")
            print(mcp_result)

            documents = retrieve_documents(
                repo_store,
                question,
            )

            print(f"\nRetrieved {len(documents)} " "repository documents.")

            repo_context = build_context(documents)

            mcp_context = str(mcp_result)

            context = f"""
Repository Code Context:

{repo_context}

GitHub MCP Context:

{mcp_context}
"""

            answer = await generate_answer(
                question,
                context,
            )

            print("\nFinal Answer:")
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
