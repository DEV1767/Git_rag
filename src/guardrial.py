import os
import requests
from dotenv import load_dotenv

from src.logger import logger

load_dotenv()

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13-20260917"


class IntentGuardrail:

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set"
            )

    def classify(self, query: str):

        questions = {

            "normal_conversation": {
                "type": "noul",
                "instructions": f"""
Determine whether the user's query is normal conversation.

Normal conversation includes:
- greetings
- casual conversation
- general questions
- questions that do not require repository information
- questions that do not require GitHub/MCP actions

User query:
{query}
"""
            },

            "repository_task": {
                "type": "noul",
                "instructions": f"""
Determine whether the user's query requires information
from the connected GitHub repository or requires performing
an operation on that repository.

Repository-related examples:
- explain the repository
- explain code
- find a function
- find a bug
- explain authentication
- show implementation
- list files
- create an issue
- create a branch
- inspect repository information

User query:
{query}
"""
            },

            "mcp_required": {
                "type": "noul",
                "instructions": f"""
Determine whether answering the user's query requires
executing a GitHub MCP tool or performing an action against
the GitHub repository.

MCP is required for operations such as:
- creating something
- modifying something
- deleting something
- listing live GitHub resources
- reading GitHub resources that are not already repository
  knowledge

Do NOT select MCP merely because the question mentions GitHub.

User query:
{query}
"""
            }
        }

        payload = {
            "model": JEV_MODEL,
            "state": {
                "query": query
            },
            "questions": questions
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:

            response = requests.post(
                JEV_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as error:

            logger.error(
                "Intent guardrail failed: %s",
                error
            )

            return {
                "route": "rag",
                "confidence": 0.0
            }

        answers = data.get("answers", {})

        normal_score = float(
            answers
            .get("normal_conversation", {})
            .get("noul", 0.0)
        )

        repository_score = float(
            answers
            .get("repository_task", {})
            .get("noul", 0.0)
        )

        mcp_score = float(
            answers
            .get("mcp_required", {})
            .get("noul", 0.0)
        )

        logger.info(
            "Intent scores | normal=%.3f | repository=%.3f | mcp=%.3f",
            normal_score,
            repository_score,
            mcp_score
        )

        # Highest intent determines route
        scores = {
            "normal": normal_score,
            "rag": repository_score,
            "mcp": mcp_score
        }

        route = max(
            scores,
            key=scores.get
        )

        return {
            "route": route,
            "confidence": scores[route],
            "scores": scores
        }