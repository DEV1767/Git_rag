from langchain_core.tools import tool


def create_issue_tools(session):

    @tool
    async def search_repo_issue(
        owner: str,
        repo: str,
        query: str = "",
    ):
        """
        Search the issues present in the repository.
        """

        result = await session.call_tool(
            "list_issues",
            {
                "owner": owner,
                "repo": repo,
                "query": query,
            },
        )

        data = []

        for content in result.content:
            if hasattr(content, "text"):
                data.append(content.text)

        return "\n".join(data)

    @tool
    async def get_the_repo_issue(
        owner: str,
        repo: str,
        issue_number: int,
    ):
        """
        Get a specific issue from the repository by issue number.
        """

        result = await session.call_tool(
            "issue_read",
            {
                "owner": owner,
                "repo": repo,
                "issue_number": issue_number,
                "method":"get"
            },
        )

        data = []

        for content in result.content:
            if hasattr(content, "text"):
                data.append(content.text)

        return "\n".join(data)

    return {
        "search_repo_issue": search_repo_issue,
        "get_the_repo_issue": get_the_repo_issue,
    }
