import re
import json


def parse_github_repo(user_input):
    user_input = user_input.strip()

    pattern = r"https?://github\.com/([^/\s]+)/([^/\s?#]+)"

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

    raise ValueError(
        "Use GitHub URL or owner/repo"
    )
    
    
