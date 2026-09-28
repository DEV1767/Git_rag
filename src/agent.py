import asyncio
import sys
from pathlib import Path
from typing import TypedDict, Any


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
    mcp_result: Any

    # Context
    repo_context: str
    mcp_context: str
    context: str

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

    selected_tool = tool["selection"]

    logger.info(
        "Selected MCP tool: %s",
        selected_tool,
    )

    return {
        "selected_tool": selected_tool
    }


# =========================================================
# 5. EXECUTE MCP TOOL
# =========================================================

async def execute_mcp_node(
    state: AgentState
):

    selected_tool = state.get(
        "selected_tool"
    )

    # If no useful tool was found,
    # don't execute anything.
    if not selected_tool or selected_tool == "NO_RELEVANT_TOOL":

        logger.warning(
            "No valid MCP tool selected."
        )

        return {
            "mcp_result": None,
            "mcp_context": (
                "No MCP information was required."
            )
        }

    arguments = {
        "owner": state["owner"],
        "repo": state["repo"],
    }

    logger.info(
        "Executing MCP tool: %s",
        selected_tool,
    )

    mcp_result = await execute_mcp_tool(
        state["session"],
        selected_tool,
        arguments,
    )

    logger.info(
        "MCP result: %s",
        mcp_result,
    )

    return {
        "mcp_result": mcp_result,
        "mcp_context": str(mcp_result),
    }


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

    # Keep existing MCP context if MCP
    # was executed.
    mcp_context = state.get(
        "mcp_context",
        "No MCP information was required."
    )

    return {
        "repo_context": repo_context,
        "mcp_context": mcp_context,
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
    # MCP → RAG
    #
    # This allows the final answer to use both:
    #
    # MCP context
    # +
    # Repository RAG context
    #
    # -----------------------------------------------------

    graph.add_edge(
        "execute_mcp",
        "retrieve_rag",
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