import asyncio
import json
from mcp_setup import (
    mcp_session,
    search_with_mcp,
)

from repo_search import build_repo_store

from retriver import (
    retrieve_documents,
    build_context,
)

from llm import Groq_model
from repo_helper import parse_github_repo
from repo_helper import extract_repository


async def generate_answer(
    question,
    context,
):
    prompt = f"""
You are a GitHub repository assistant.

Answer the user's question using only the repository context provided below.

USER QUESTION:
{question}

REPOSITORY CONTEXT:
{context}

Rules:
- Answer only from the repository context.
- Do not invent information.
- If the answer cannot be found in the context, say:
"I could not find the answer in the repository."
- Mention the relevant file names when useful.
- Explain the code clearly and concisely.

ANSWER:
"""

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    return content


async def main():

    print("\n=================================")
    print("       GitHub RAG Agent")
    print("=================================")

    user_input = input("\nEnter GitHub repository " "(owner/repo or URL): ")

    owner, repo = parse_github_repo(user_input)

    print("\nRequested repository:")
    print(f"Owner : {owner}")
    print(f"Repo  : {repo}")

    async with mcp_session() as session:

        print("\nMCP connected.")

        query = f"Find the GitHub repository " f"{owner}/{repo}"

        (
            tool_name,
            arguments,
            result_text,
        ) = await search_with_mcp(
            session,
            query,
        )

        print("\nSelected MCP tool:")
        print(tool_name)

        if tool_name != "search_repositories":
            print("\nThe agent did not select " "search_repositories.")
            return

        repository = extract_repository(
            result_text,
            owner,
            repo,
        )

        if repository is None:
            print("\nRepository not found.")
            return

        print("\n=================================")
        print("Repository found")
        print("=================================")

        print(f"Name       : " f"{repository.get('full_name')}")

        print(f"Description: " f"{repository.get('description')}")

        print(f"URL        : " f"{repository.get('html_url')}")

        print("\nBuilding repository RAG...")

        repo_store = await build_repo_store(
            session,
            owner,
            repo,
        )

        print("\nRepository RAG created successfully.")

        while True:

            question = input("\nWhat do you want to know " "about this repository? ")

            if question.lower() in {
                "exit",
                "quit",
                "q",
            }:
                break

            documents = retrieve_documents(
                repo_store,
                question,
                k=5,
            )

            if not documents:
                print("\nNo relevant repository " "content found.")
                continue

            context = build_context(documents)

            print("\nGenerating answer...")

            answer = await generate_answer(
                question,
                context,
            )

            print("\n=================================")
            print("Final Answer")
            print("=================================")
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
