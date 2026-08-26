import asyncio
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from src.mcp_setup import (
    mcp_session,
    execute_mcp_tool,
)

from src.repo_search import build_repo_store

from src.retriver import (
    retrieve_documents,
    build_context,
)

from src.llm import Groq_model

from src.repo_helper import parse_github_repo

from src.qdrant_setup import build_tool_store

from src.prompt_helper import retriever_prompt

from src.tools.tool_selector import select_tool


class AgentState(TypedDict, total=False):
    question: str
    owner: str
    repo: str
    tool_store: Any
    repo_store: Any
    session: Any
    selected_tool: str
    mcp_result: Any
    repo_context: str
    mcp_context: str
    context: str
    answer: str


async def select_tool_node(state: AgentState):
    tool = await select_tool(
        state["question"],
        state["session"],
        state["tool_store"],
    )

    selected_tool = tool["selection"]

    print("\nSelected tool:")
    print(selected_tool)

    return {"selected_tool": selected_tool}


def route_after_tool_selection(state: AgentState):

    if state["selected_tool"] == "NO_RELEVANT_TOOL":
        return "rag"

    return "mcp"


async def execute_mcp_node(state: AgentState):

    arguments = {
        "owner": state["owner"],
        "repo": state["repo"],
    }

    mcp_result = await execute_mcp_tool(
        state["session"],
        state["selected_tool"],
        arguments,
    )

    print("\nMCP Result:")
    print(mcp_result)

    return {
        "mcp_result": mcp_result,
        "mcp_context": str(mcp_result),
    }


async def retrieve_rag_node(state: AgentState):

    documents = retrieve_documents(
        state["repo_store"],
        state["question"],
    )

    print(f"\nRetrieved {len(documents)} " "repository documents.")

    repo_context = build_context(documents)

    return {
        "repo_context": repo_context,
        "mcp_context": "No MCP information was required.",
    }


async def generate_answer_node(state: AgentState):

    context = f"""
Repository Code Context:

{state.get("repo_context", "")}

GitHub MCP Context:

{state.get("mcp_context", "No MCP information was required.")}
"""

    prompt = retriever_prompt.invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    return {
        "context": context,
        "answer": content,
    }


def build_graph():

    graph = StateGraph(AgentState)

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

    graph.add_edge(
        START,
        "select_tool",
    )

    graph.add_conditional_edges(
        "select_tool",
        route_after_tool_selection,
        {
            "mcp": "execute_mcp",
            "rag": "retrieve_rag",
        },
    )

    graph.add_edge(
        "execute_mcp",
        "retrieve_rag",
    )

    graph.add_edge(
        "retrieve_rag",
        "generate_answer",
    )

    graph.add_edge(
        "generate_answer",
        END,
    )

    return graph.compile()


async def main():

    repository_url = input("Enter the repository url :\n")

    owner, repo = parse_github_repo(repository_url)

    tool_store = build_tool_store()

    async with mcp_session() as session:

        print("\nBuilding repository RAG store...")

        repo_store = await build_repo_store(
            session,
            owner,
            repo,
        )

        app = build_graph()

        while True:

            question = input("\nEnter your Query " "(or type 'exit'):\n")

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

            result = await app.ainvoke(initial_state)

            print("\nFinal Answer:")
            print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
