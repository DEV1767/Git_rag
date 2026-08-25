import asyncio

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

from repo_helper import (
    parse_github_repo,
    extract_repository,
)

from prompt_helper import (
    agent_prompt,
    retriever_prompt,
    issue_prompt,
)

from tools.github_helper import create_issue_tools


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

    user_input = input("\nEnter GitHub repository " "(owner/repo or URL): ")

    owner, repo = parse_github_repo(user_input)

    # ==========================================================
    # ONE MCP SESSION
    # ==========================================================

    async with mcp_session() as session:

        print("\nMCP connected.")

        # ======================================================
        # CREATE ISSUE TOOLS USING SAME MCP SESSION
        # ======================================================

        issue_tools = create_issue_tools(session)

        search_issue = issue_tools["search_repo_issue"]

        get_issue = issue_tools["get_the_repo_issue"]

        # ======================================================
        # BIND ISSUE TOOLS TO LLM
        # ======================================================

        llm_with_tools = Groq_model.bind_tools(
            [
                search_issue,
                get_issue,
            ]
        )

        # ======================================================
        # FIND REPOSITORY
        # ======================================================

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

        print(f"Name        : " f"{repository.get('full_name')}")

        print(f"Description : " f"{repository.get('description')}")

        print(f"URL         : " f"{repository.get('html_url')}")

        # ======================================================
        # BUILD REPOSITORY RAG
        # ======================================================

        print("\nBuilding repository RAG...")

        repo_store = await build_repo_store(
            session,
            owner,
            repo,
        )

        print("\nRepository RAG created successfully.")

        # ======================================================
        # QUESTION LOOP
        # ======================================================

        while True:

            question = input("\nWhat do you want to know " "about this repository? ")

            if question.lower() in {
                "exit",
                "quit",
                "q",
            }:
                break

            # ==================================================
            # AGENT / TOOL ROUTER
            # ==================================================

            routing_prompt = agent_prompt.invoke(
                {
                    "question": question,
                }
            )

            response = await llm_with_tools.ainvoke(routing_prompt)

            # ==================================================
            # ISSUE TOOL SELECTED
            # ==================================================

            if response.tool_calls:

                tool_messages = []

                for tool_call in response.tool_calls:

                    tool_name = tool_call["name"]

                    tool_args = dict(tool_call["args"])

                    # ------------------------------------------
                    # Repository is already known
                    # ------------------------------------------

                    tool_args["owner"] = owner
                    tool_args["repo"] = repo

                    # ------------------------------------------
                    # Search issues
                    # ------------------------------------------

                    if tool_name == "search_repo_issue":

                        tool_result = await search_issue.ainvoke(tool_args)

                    # ------------------------------------------
                    # Get specific issue
                    # ------------------------------------------

                    elif tool_name == "get_the_repo_issue":

                        tool_result = await get_issue.ainvoke(tool_args)

                    else:

                        continue

                    tool_messages.append(
                        {
                            "role": "tool",
                            "content": str(tool_result),
                            "tool_call_id": (tool_call["id"]),
                        }
                    )

                # ==================================================
                # BUILD ISSUE CONTEXT
                # ==================================================

                issue_context = "\n\n".join(
                    message["content"] for message in tool_messages
                )

                # ==================================================
                # FINAL ISSUE ANSWER
                # ==================================================

                issue_answer_prompt = issue_prompt.invoke(
                    {
                        "question": question,
                        "issue_context": issue_context,
                    }
                )

                final_response = await Groq_model.ainvoke(issue_answer_prompt)

                content = final_response.content

                if isinstance(content, list):

                    content = "".join(str(x) for x in content)

                print("\n=================================")
                print("Final Answer")
                print("=================================")

                print(content)

                continue

            # ==================================================
            # NORMAL REPOSITORY RAG
            # ==================================================

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
