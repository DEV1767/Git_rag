import asyncio
import json
import re

from mcp_setup import (
    mcp_session,
    search_with_mcp,
)

from repo_search import build_repo_store


def parse_github_repo(user_input):

    user_input = user_input.strip()

    pattern = r"https?://github\.com/" r"([^/\s]+)/" r"([^/\s?#]+)"

    match = re.search(
        pattern,
        user_input,
    )

    if match:

        return (
            match.group(1),
            match.group(2).replace(
                ".git",
                "",
            ),
        )

    if "/" in user_input:

        parts = user_input.split(
            "/",
            1,
        )

        owner = parts[0].strip()
        repo = parts[1].strip()

        if owner and repo:

            return owner, repo

    raise ValueError("Use GitHub URL or owner/repo")


def extract_repository(
    result_text,
    owner,
    repo,
):

    try:

        data = json.loads(result_text)

    except json.JSONDecodeError:

        print("\nCould not parse MCP result.")

        print(result_text)

        return None

    items = data.get(
        "items",
        [],
    )

    wanted = (f"{owner}/{repo}").lower()

    for item in items:

        full_name = item.get(
            "full_name",
            "",
        ).lower()

        if full_name == wanted:

            return item

    return None


async def main():

    print("\n=================================")

    print("       GitHub RAG Agent")

    print("=================================")

    user_input = input("\nEnter GitHub repository " "(owner/repo or URL): ")

    owner, repo = parse_github_repo(user_input)

    print(f"\nRequested repository:")

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

        question = input("\nWhat do you want to know " "about this repository? ")

        documents = repo_store.similarity_search(
            question,
            k=5,
        )

        print("\nRelevant repository files:")

        for doc in documents:

            print(
                "-",
                doc.metadata.get(
                    "path",
                    "unknown",
                ),
            )

        print("\nRepository chunks retrieved:")

        print(len(documents))


if __name__ == "__main__":

    asyncio.run(main())
