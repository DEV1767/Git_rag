import asyncio

from mcp_setup import mcp_session, execute_mcp_tool

from repo_search import build_repo_store


from llm import Groq_model

from repo_helper import (
    parse_github_repo,
    extract_repository,
)

from qdrant_setup import build_tool_store

from prompt_helper import retriever_prompt, tool_selection_prompt

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

        while True:

            question = input("\nEnter your Query (or type 'exit'):\n")

            if question.lower() == "exit":
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

            result = await execute_mcp_tool(
                session,
                selected_tool,
                arguments,
            )

            print("\nMCP Result:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
