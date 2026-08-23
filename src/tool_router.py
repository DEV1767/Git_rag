TOOL_GROUPS = {
    "repository": [
        "search_repositories",
        "search_code",
        "get_file_contents",
        "list_branches",
    ],
    "issues": [
        "issue_read",
        "list_issues",
        "search_issues",
    ],
    "pull_requests": [
        "pull_request_read",
        "list_pull_requests",
        "search_pull_requests",
    ],
    "commits": [
        "get_commit",
        "list_commits",
        "search_commits",
    ],
    "account": [
        "get_me",
    ],
}


def select_tools(all_tools, group):
    allowed = TOOL_GROUPS.get(group, [])

    return [tool for tool in all_tools if tool.name in allowed]
