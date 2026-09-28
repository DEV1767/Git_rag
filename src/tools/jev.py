import os
import requests

from dotenv import load_dotenv

from src.logger import logger


load_dotenv()


JEV_URL = "https://openrouter.ai/api/alpha/decisions"

JEV_MODEL = "typesafe/jev-1.13-20260917"


class JEVToolSelector:

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set"
            )

    def rerank(
        self,
        query,
        tools,
        top_k=1,
    ):

        if not tools:
            return []

        # -----------------------------------------
        # Build JEV questions
        # -----------------------------------------

        questions = {}

        for index, tool in enumerate(tools):

            document = tool["document"]

            tool_name = document.metadata.get(
                "name",
                "",
            )

            description = (
                document.page_content or ""
            )

            questions[f"tool_{index}"] = {
                "type": "noul",
                "instructions": (
                    "Is this MCP tool relevant "
                    "for answering the user's query?\n\n"

                    f"User query:\n"
                    f"{query}\n\n"

                    f"Tool name:\n"
                    f"{tool_name}\n\n"

                    f"Tool description:\n"
                    f"{description}"
                ),
            }

        # -----------------------------------------
        # State sent to JEV
        # -----------------------------------------

        state = {
            "query": query,
            "tools": [
                {
                    "name": tool["document"]
                    .metadata
                    .get("name", ""),

                    "description": (
                        tool["document"]
                        .page_content or ""
                    ),
                }
                for tool in tools
            ],
        }

        # -----------------------------------------
        # Request headers
        # -----------------------------------------

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        # -----------------------------------------
        # Request payload
        # -----------------------------------------

        payload = {
            "model": JEV_MODEL,
            "state": state,
            "questions": questions,
        }

        logger.info(
            "Sending %d tools to JEV",
            len(tools),
        )

        # -----------------------------------------
        # Call JEV
        # -----------------------------------------

        try:

            response = requests.post(
                JEV_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as error:

            logger.error(
                "JEV request failed: %s",
                error,
            )

            return []

        # -----------------------------------------
        # Read JEV answers
        # -----------------------------------------

        answers = data.get(
            "answers",
            {},
        )

        ranked = []

        # -----------------------------------------
        # Attach JEV scores
        # -----------------------------------------

        for index, tool in enumerate(tools):

            answer = answers.get(
                f"tool_{index}",
                {},
            )

            score = float(
                answer.get(
                    "noul",
                    0.0,
                )
            )

            tool["jev_score"] = score

            ranked.append(tool)

        # -----------------------------------------
        # Sort by JEV score
        # -----------------------------------------

        ranked.sort(
            key=lambda x: x["jev_score"],
            reverse=True,
        )

        # -----------------------------------------
        # Logging
        # -----------------------------------------

        logger.info(
            "========== JEV TOOL RANKING =========="
        )

        for tool in ranked:

            logger.info(
                "Tool=%s | Qdrant=%.4f | JEV=%.4f",

                tool["document"]
                .metadata
                .get("name"),

                tool.get(
                    "qdrant_score",
                    0.0,
                ),

                tool.get(
                    "jev_score",
                    0.0,
                ),
            )

        logger.info(
            "======================================"
        )

        return ranked[:top_k]