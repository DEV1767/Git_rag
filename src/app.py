from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException

from src.repo_helper import parse_github_repo
from src.mcp_setup import mcp_session

app = FastAPI(
    title="GitHub Agentic RAG",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    repository_url: str
    question: str


@app.get("/")
async def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    from src.agent import build_graph
    from src.qdrant_setup import build_tool_store
    from src.repo_search import build_repo_store

    if not request.question.strip():
        return {
            "repository": request.repository_url,
            "question": request.question,
            "answer": "Please enter a question.",
        }

    try:
        owner, repo = parse_github_repo(request.repository_url)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Use a GitHub URL or owner/repo, for example: octocat/Hello-World",
        ) from error

    try:
        tool_store = build_tool_store()
    except ResponseHandlingException as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant is unavailable. Check QDRANT_URL, QDRANT_API_KEY, "
                "and network access, then try again."
            ),
        ) from error

    try:
        async with mcp_session() as session:

            repo_store = await build_repo_store(
                session,
                owner,
                repo,
            )

            graph = build_graph()

            result = await graph.ainvoke(
                {
                    "question": request.question,
                    "owner": owner,
                    "repo": repo,
                    "tool_store": tool_store,
                    "repo_store": repo_store,
                    "session": session,
                }
            )

            return {
                "repository": f"{owner}/{repo}",
                "question": request.question,
                "selected_tool": result.get("selected_tool"),
                "answer": result["answer"],
            }
    except ExceptionGroup as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub MCP closed before initialization. Confirm Docker Desktop "
                "is running, the GitHub MCP image can start, and "
                "GITHUB_PERSONAL_ACCESS_TOKEN is valid."
            ),
        ) from error
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub MCP could not start. Start Docker Desktop and verify "
                "that the GitHub MCP server image is available, then try again."
            ),
        ) from error
    except ResponseHandlingException as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Qdrant became unavailable while building repository context. "
                "Check the Qdrant service and try again."
            ),
        ) from error
