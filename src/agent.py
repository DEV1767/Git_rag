import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict, Any, Dict, List


if __package__ in {None, ""}:
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent)
    )



from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.mcp_setup import (
    mcp_session,
    execute_mcp_tool,
)

from src.repo_search import build_repo_store

from src.retriver import (
    retrieve_hybrid_documents,
    build_context,
)

from src.llm import Groq_model

from src.repo_helper import parse_github_repo

from src.qdrant_setup import build_tool_store

from src.prompt_helper import retriever_prompt

from src.intent_router import IntentRouter

from src.tools.tool_selector import select_tool

from src.logger import logger


# =========================================================
# AGENT STATE
# =========================================================

class AgentState(TypedDict, total=False):

    # User query
    question: str

    # Repository
    owner: str
    repo: str

    # Intent routing
    intent: str
    intent_confidence: float

    # Stores
    tool_store: Any
    repo_store: Any
    embeddings: Any

    # MCP
    session: Any
    selected_tool: str
    tool_schema: str
    tool_arguments: Dict[str, Any]
    mcp_result: Any

    # Context & Sources
    repo_context: str
    mcp_context: str
    context: str
    sources: List[str]

    # Final answer
    answer: str


# =========================================================
# 1. INTENT NODE
# =========================================================

async def intent_node(state: AgentState):

    router = IntentRouter()

    result = router.route(
        state["question"]
    )

    logger.info(
        "Intent Router → %s | confidence=%.4f",
        result["route"],
        result["confidence"],
    )

    return {
        "intent": result["route"],
        "intent_confidence": result["confidence"],
    }


# =========================================================
# 2. ROUTE AFTER INTENT
# =========================================================

def route_after_intent(state: AgentState):

    intent = state.get("intent", "rag")

    logger.info(
        "Routing query to: %s",
        intent,
    )

    if intent == "normal":
        return "normal"

    if intent == "rag":
        return "rag"

    if intent == "mcp":
        return "mcp"

    # Safe fallback
    logger.warning(
        "Unknown intent '%s'. Falling back to RAG.",
        intent,
    )

    return "rag"


# =========================================================
# 3. NORMAL CONVERSATION NODE
# =========================================================

async def normal_conversation_node(
    state: AgentState
):

    question = state["question"]

    logger.info(
        "Handling as normal conversation."
    )

    prompt = f"""
You are a helpful conversational assistant.

Answer the user's question naturally and concisely.

This is a normal conversation, so do not use
GitHub repository context or MCP tools.

User question:
{question}
"""

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            str(x)
            for x in content
        )

    return {
        "answer": content
    }


# =========================================================
# 4. MCP TOOL SELECTION NODE
# =========================================================

async def select_tool_node(
    state: AgentState
):

    logger.info(
        "Selecting MCP tool..."
    )

    tool = await select_tool(
        state["question"],
        state["session"],
        state["tool_store"],
    )

    selected_tool = tool.get("selection", "NO_RELEVANT_TOOL")
    tool_schema = tool.get("schema", "{}")
    tool_score = tool.get("score", 0.0)

    # Discard if below minimal confidence or explicitly irrelevant
    if not selected_tool or selected_tool == "NO_RELEVANT_TOOL":
        logger.info("No relevant MCP tool selected.")
        return {
            "selected_tool": "NO_RELEVANT_TOOL",
            "tool_schema": "{}",
        }

    logger.info(
        "Selected MCP tool: %s | JEV score=%.4f",
        selected_tool,
        tool_score,
    )

    return {
        "selected_tool": selected_tool,
        "tool_schema": tool_schema,
    }


# =========================================================
# 5. EXECUTE MCP TOOL
# =========================================================

def format_mcp_result(mcp_result) -> str:
    """Extract clean, readable text from MCP CallToolResult."""
    if not mcp_result:
        return "No MCP information was returned."

    if hasattr(mcp_result, "content") and isinstance(mcp_result.content, list):
        text_parts = []
        for item in mcp_result.content:
            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)
        if text_parts:
            return "\n\n".join(text_parts)

    return str(mcp_result)


async def prepare_tool_arguments(
    question: str,
    tool_name: str,
    tool_schema: str,
    owner: str,
    repo: str,
) -> Dict[str, Any]:
    """
    Use LLM to extract dynamic tool parameters from user question
    guided by the tool's JSON input schema.
    """
    prompt = f"""You are a GitHub MCP tool argument generator.
The user wants to execute the GitHub tool: '{tool_name}'.

Tool Input Schema:
{tool_schema}

Repository Identity:
owner: {owner}
repo: {repo}

User Query/Request:
{question}

Instructions:
1. Extract or infer the parameters required by '{tool_name}' from the user request.
2. Always ensure 'owner' is set to '{owner}' and 'repo' is set to '{repo}' if applicable.
3. If an issue number, pull request number, branch name, or query string is mentioned, map it to the corresponding schema property.
4. Output ONLY a valid JSON object of arguments. Do not include markdown code fences (```), explanations, or extra commentary.

Example:
{{"owner": "{owner}", "repo": "{repo}"}}
"""
    try:
        response = await Groq_model.ainvoke(prompt)
        text = response.content
        if isinstance(text, list):
            text = "".join(str(x) for x in text)
        clean_text = text.strip().replace("```json", "").replace("```", "").strip()
        args = json.loads(clean_text)
        if isinstance(args, dict):
            args.setdefault("owner", owner)
            args.setdefault("repo", repo)
            return args
    except Exception as e:
        logger.warning("Could not parse LLM tool arguments, using owner/repo fallback: %s", e)

    return {"owner": owner, "repo": repo}


async def execute_mcp_node(
    state: AgentState
):

    selected_tool = state.get(
        "selected_tool"
    )

    if not selected_tool or selected_tool == "NO_RELEVANT_TOOL":
        logger.warning(
            "No valid MCP tool selected. Skipping MCP execution."
        )
        return {
            "mcp_result": None,
            "mcp_context": (
                "No MCP information was required."
            )
        }

    tool_schema = state.get("tool_schema", "{}")
    arguments = await prepare_tool_arguments(
        question=state["question"],
        tool_name=selected_tool,
        tool_schema=tool_schema,
        owner=state["owner"],
        repo=state["repo"],
    )

    logger.info(
        "Executing MCP tool: %s with arguments: %s",
        selected_tool,
        arguments,
    )

    try:
        mcp_result = await execute_mcp_tool(
            state["session"],
            selected_tool,
            arguments,
        )
    except Exception as error:
        logger.error(
            "Error executing MCP tool %s: %s",
            selected_tool,
            error,
        )
        mcp_result = f"Error executing tool {selected_tool}: {error}"

    formatted_context = format_mcp_result(mcp_result)

    logger.info(
        "MCP execution complete. Output preview: %s",
        formatted_context[:250],
    )

    return {
        "tool_arguments": arguments,
        "mcp_result": mcp_result,
        "mcp_context": formatted_context,
    }


def route_after_mcp(state: AgentState):
    """
    Decide whether to also retrieve repository code after MCP tool execution.
    - If no MCP tool was executed, fallback to RAG.
    - If the user query is purely an MCP task (issues, PRs, branches, releases) and has no code intent,
      route directly to generate_answer.
    - If the user query asks about code files, functions, or implementation, also retrieve code RAG.
    """
    selected_tool = state.get("selected_tool")
    if not selected_tool or selected_tool == "NO_RELEVANT_TOOL":
        return "retrieve_rag"

    question = state.get("question", "").lower()
    code_intent_words = [
        "code", "file", "function", "implement", "class", "method",
        "where", "route", "controller", "how", "middleware", "logic"
    ]

    if any(word in question for word in code_intent_words):
        logger.info("Hybrid query detected: routing to repository RAG for code context.")
        return "retrieve_rag"

    logger.info("Pure MCP query detected: routing directly to answer generation.")
    return "generate_answer"


# =========================================================
# 6. RAG RETRIEVAL NODE
# =========================================================

async def retrieve_rag_node(
    state: AgentState
):

    collection_name = (
        f"github_{state['owner']}_{state['repo']}"
    ).replace("/", "_")

    logger.info(
        "Starting repository RAG retrieval..."
    )

    documents = retrieve_hybrid_documents(
        vectorstore=state["repo_store"],
        query=state["question"],
        collection_name=collection_name,
        k=30,
    )

    logger.info(
        "Retrieved %d unique hybrid documents.",
        len(documents),
    )

    repo_context = build_context(
        documents
    )

    # Extract unique source paths
    sources = []
    seen = set()
    for doc_item in documents:
        doc = doc_item.get("document")
        if doc and hasattr(doc, "metadata"):
            p = doc.metadata.get("path")
            if p and p not in seen:
                seen.add(p)
                sources.append(p)

    # Keep existing MCP context if MCP
    # was executed.
    mcp_context = state.get(
        "mcp_context",
        "No MCP information was required."
    )

    return {
        "repo_context": repo_context,
        "mcp_context": mcp_context,
        "sources": sources,
    }


# =========================================================
# 7. GENERATE ANSWER
# =========================================================

async def generate_answer_node(
    state: AgentState
):

    repo_context = state.get(
        "repo_context",
        ""
    )

    mcp_context = state.get(
        "mcp_context",
        "No MCP information was required."
    )

    context = f"""
Repository Code Context:

{repo_context}

GitHub MCP Context:

{mcp_context}
"""

    prompt = retriever_prompt.invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )

    logger.info(
        "Generating final answer..."
    )

    response = await Groq_model.ainvoke(
        prompt
    )

    content = response.content

    if isinstance(content, list):
        content = "".join(
            str(x)
            for x in content
        )

    return {
        "context": context,
        "answer": content,
    }


# =========================================================
# 8. BUILD LANGGRAPH
# =========================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    graph.add_node(
        "intent",
        intent_node,
    )

    graph.add_node(
        "normal_conversation",
        normal_conversation_node,
    )

    graph.add_node(
        "select_tool",
        select_tool_node,
    )

    graph.add_node(
        "execute_mcp",
        execute_mcp_node,
    )

    graph.add_node(
        "retrieve_rag",
        retrieve_rag_node,
    )

    graph.add_node(
        "generate_answer",
        generate_answer_node,
    )

    # -----------------------------------------------------
    # START → INTENT
    # -----------------------------------------------------

    graph.add_edge(
        START,
        "intent",
    )

    # -----------------------------------------------------
    # INTENT → NORMAL / RAG / MCP
    # -----------------------------------------------------

    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "normal": "normal_conversation",
            "rag": "retrieve_rag",
            "mcp": "select_tool",
        },
    )

    # -----------------------------------------------------
    # NORMAL → END
    # -----------------------------------------------------

    graph.add_edge(
        "normal_conversation",
        END,
    )

    # -----------------------------------------------------
    # MCP TOOL SELECTION → EXECUTION
    # -----------------------------------------------------

    graph.add_edge(
        "select_tool",
        "execute_mcp",
    )

    # -----------------------------------------------------
    # MCP → RAG or FINAL LLM (Dynamic Routing)
    #
    # If the user query is a pure MCP task (e.g. list issues, list PRs),
    # route directly to generate_answer.
    # If it is a hybrid query (mentions code/implementation) or no tool
    # was selected, retrieve repository code context.
    # -----------------------------------------------------

    graph.add_conditional_edges(
        "execute_mcp",
        route_after_mcp,
        {
            "retrieve_rag": "retrieve_rag",
            "generate_answer": "generate_answer",
        },
    )

    # -----------------------------------------------------
    # RAG → FINAL LLM
    # -----------------------------------------------------

    graph.add_edge(
        "retrieve_rag",
        "generate_answer",
    )

    # -----------------------------------------------------
    # FINAL LLM → END
    # -----------------------------------------------------

    graph.add_edge(
        "generate_answer",
        END,
    )

    return graph.compile()


# =========================================================
# 9. MAIN
# =========================================================

async def main():

    repository_url = input(
        "Enter the repository url :\n"
    )

    owner, repo = parse_github_repo(
        repository_url
    )

    # -----------------------------------------------------
    # Build Tool Store
    # -----------------------------------------------------

    tool_store = build_tool_store()

    # -----------------------------------------------------
    # MCP session
    # -----------------------------------------------------

    async with mcp_session() as session:

        # -------------------------------------------------
        # Build Repository RAG Store
        # -------------------------------------------------

        logger.info(
            "Building repository RAG store..."
        )

        repo_store = await build_repo_store(
            session,
            owner,
            repo,
        )

        # -------------------------------------------------
        # Build Agent Graph
        # -------------------------------------------------

        app = build_graph()

        # -------------------------------------------------
        # Conversation Loop
        # -------------------------------------------------

        while True:

            question = input(
                "\nEnter your Query "
                "(or type 'exit'):\n"
            )

            if question.lower().strip() == "exit":
                break

            initial_state = {

                "question": question,

                "owner": owner,

                "repo": repo,

                "tool_store": tool_store,

                "repo_store": repo_store,

                "session": session,
            }

            # -------------------------------------------------
            # Run Agent
            # -------------------------------------------------

            result = await app.ainvoke(
                initial_state
            )

            print(
                "\nFinal Answer:"
            )

            print(
                result.get(
                    "answer",
                    "No answer generated."
                )
            )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )