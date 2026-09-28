from langchain_core.documents import Document

from src.logger import logger
from src.tools.jev import JEVToolSelector


async def discover_mcp_tools(session):
    """
    Discover all tools exposed by the MCP server
    and convert them into LangChain Documents.
    """

    result = await session.list_tools()

    documents = []

    for tool in result.tools:

        document = Document(
            page_content=tool.description or "",
            metadata={
                "name": tool.name,
                "source": "github_mcp",
                "input_schema": str(tool.inputSchema),
            },
        )

        documents.append(document)

    logger.info(
        "Discovered %d MCP tools",
        len(documents),
    )

    return documents


async def add_mcp_to_store(session, tool_store):
    """
    Discover MCP tools and add them to the Tool Store.
    """

    documents = await discover_mcp_tools(session)

    if not documents:
        logger.warning(
            "No MCP tools found."
        )
        return []

    tool_store.add_documents(documents)

    logger.info(
        "Added %d MCP tools to Tool Store.",
        len(documents),
    )

    return documents


def deduplicate_tools(results):
    """
    Remove duplicate tools returned by Qdrant.

    Keep one document for each unique MCP tool.
    """

    unique_tools = {}

    for document, score in results:

        tool_name = document.metadata.get(
            "name"
        )

        if not tool_name:
            continue

        if tool_name not in unique_tools:

            unique_tools[tool_name] = {
                "document": document,
                "qdrant_score": float(score),
            }

    return list(
        unique_tools.values()
    )


async def rank_tools(query, results):
    """
    Rank retrieved MCP tools using JEV.
    """

    tools = deduplicate_tools(results)

    logger.info(
        "Unique Tool Store candidates: %d",
        len(tools),
    )

    if not tools:
        return []

    logger.info(
        "========== TOOL CANDIDATES =========="
    )

    for tool in tools:

        document = tool["document"]

        logger.info(
            "Candidate: %s | Qdrant=%.4f",
            document.metadata.get("name"),
            tool["qdrant_score"],
        )

    logger.info(
        "====================================="
    )

    # -----------------------------------------
    # JEV
    # -----------------------------------------

    selector = JEVToolSelector()

    ranked_tools = selector.rerank(
        query=query,
        tools=tools,
        top_k=1,
    )

    return ranked_tools


async def select_tool(
    query: str,
    session,
    tool_store,
):
    """
    Select the most relevant MCP tool.

    Flow:

    1. Search Tool Store broadly.
    2. Remove duplicate tools.
    3. Rank candidates using JEV.
    4. If no candidates exist, discover MCP tools.
    5. Add discovered tools to Tool Store.
    6. Search again.
    7. Rank again using JEV.
    8. Return selected tool.
    """

    # ==========================================
    # 1. Broad Tool Store search
    # ==========================================

    results = tool_store.similarity_search_with_score(
        query,
        k=30,
    )

    logger.info(
        "Tool Store returned %d candidates",
        len(results),
    )

    # ==========================================
    # 2. JEV ranking
    # ==========================================

    ranked_tools = await rank_tools(
        query,
        results,
    )

    # ==========================================
    # 3. JEV selected a tool
    # ==========================================

    if ranked_tools:

        best_tool = ranked_tools[0]

        tool_name = (
            best_tool["document"]
            .metadata
            .get("name")
        )

        score = best_tool.get(
            "jev_score",
            0.0,
        )

        logger.info(
            "JEV selected tool: %s | score=%.4f",
            tool_name,
            score,
        )

        return {
            "source": "jev",
            "selection": tool_name,
            "tools": ranked_tools,
            "score": score,
        }

    # ==========================================
    # 4. No candidates
    # ==========================================

    logger.info(
        "No useful Tool Store candidates."
    )

    logger.info(
        "Discovering tools from GitHub MCP..."
    )

    mcp_documents = await add_mcp_to_store(
        session,
        tool_store,
    )

    # ==========================================
    # 5. No MCP tools
    # ==========================================

    if not mcp_documents:

        logger.warning(
            "No MCP tools available."
        )

        return {
            "source": "mcp",
            "selection": "NO_RELEVANT_TOOL",
            "tools": [],
        }

    # ==========================================
    # 6. Search Tool Store again
    # ==========================================

    results = tool_store.similarity_search_with_score(
        query,
        k=30,
    )

    logger.info(
        "Candidates after MCP discovery: %d",
        len(results),
    )

    # ==========================================
    # 7. JEV ranking again
    # ==========================================

    ranked_tools = await rank_tools(
        query,
        results,
    )

    # ==========================================
    # 8. Still nothing useful
    # ==========================================

    if not ranked_tools:

        logger.warning(
            "JEV could not select a relevant tool."
        )

        return {
            "source": "mcp_discovery",
            "selection": "NO_RELEVANT_TOOL",
            "tools": [],
        }

    # ==========================================
    # 9. Final selected tool
    # ==========================================

    best_tool = ranked_tools[0]

    tool_name = (
        best_tool["document"]
        .metadata
        .get("name")
    )

    score = best_tool.get(
        "jev_score",
        0.0,
    )

    logger.info(
        "JEV selected after discovery: %s | score=%.4f",
        tool_name,
        score,
    )

    return {
        "source": "mcp_discovery",
        "selection": tool_name,
        "tools": ranked_tools,
        "score": score,
    }