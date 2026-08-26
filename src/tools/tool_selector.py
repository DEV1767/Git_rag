from src.prompt_helper import tool_selection_prompt
from langchain_core.documents import Document
from src.llm import Groq_model


async def discover_mcp_tools(session):
    """
    Discover all tools exposed by the MCP server
    and convert them into LangChain Documents.
    """

    result = await session.list_tools()

    documents = []

    for tool in result.tools:
        tool_name = tool.name
        description = tool.description or ""

        document = Document(
            page_content=description,
            metadata={
                "name": tool_name,
                "source": "github_mcp",
                "input_schema": str(tool.inputSchema),
            },
        )

        documents.append(document)

    return documents


async def add_mcp_to_store(session, tool_store):
    """
    Discover MCP tools and add them to the Qdrant Tool Store.
    """

    documents = await discover_mcp_tools(session)

    if not documents:
        print("No MCP tools found.")
        return []

    tool_store.add_documents(documents)

    print(f"Added {len(documents)} MCP tools " "to Tool Store.")

    return documents


async def select_tool(
    query: str,
    session,
    tool_store,
):
    """
    Select the most relevant MCP tool.

    Flow:

    1. Search the existing Qdrant Tool Store.
    2. Ask Groq to select the relevant tool.
    3. If no relevant tool exists:
       - Discover tools from MCP.
       - Add them to Qdrant.
       - Search Qdrant again.
       - Ask Groq to select again.

    MCP discovery is completely handled inside
    this module.
    """

    results = tool_store.similarity_search_with_score(
        query,
        k=5,
    )

    print("\nTool Store candidates:")

    if results:
        for document, score in results:
            print(f"{document.metadata.get('name')} " f"(score={score})")
    else:
        print("No candidates found.")

    tools = "\n\n".join(
        f"Tool: {document.metadata.get('name')}\n"
        f"Description: {document.page_content}"
        for document, score in results
    )

    prompt = tool_selection_prompt.invoke(
        {
            "question": query,
            "tools": tools,
        }
    )

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    content = str(content).strip()

    if "NO_RELEVANT_TOOL" not in content:

        print(f"\nSelected tool from Tool Store: " f"{content}")

        return {
            "source": "tool_store",
            "selection": content,
            "tools": results,
        }

    print("\nNo relevant tool found in Tool Store.")

    print("Discovering tools from GitHub MCP...")

    mcp_documents = await add_mcp_to_store(
        session,
        tool_store,
    )

    if not mcp_documents:
        print("No MCP tools available.")

        return {
            "source": "mcp",
            "selection": "No MCP tools available.",
            "tools": [],
        }

    results = tool_store.similarity_search_with_score(
        query,
        k=5,
    )

    print("\nTool Store candidates after " "MCP discovery:")

    if results:
        for document, score in results:
            print(f"{document.metadata.get('name')} " f"(score={score})")
    else:
        print("No candidates found after discovery.")

    tools = "\n\n".join(
        f"Tool: {document.metadata.get('name')}\n"
        f"Description: {document.page_content}"
        for document, score in results
    )

    prompt = tool_selection_prompt.invoke(
        {
            "question": query,
            "tools": tools,
        }
    )

    response = await Groq_model.ainvoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    content = str(content).strip()

    print(f"\nSelected tool after MCP discovery: " f"{content}")

    return {
        "source": "mcp_discovery",
        "selection": content,
        "tools": results,
    }


