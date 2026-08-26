from fastapi import FastAPI
from pydantic import BaseModel

from src.repo_helper import parse_github_repo
from src.qdrant_setup import build_tool_store
from src.repo_search import build_repo_store
from src.mcp_setup import mcp_session
from src.agent import build_graph

app = FastAPI(
    title="GitHub Agentic RAG",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    repository_url: str
    question: str


@app.get('/ask')
def ask():
    return "Server is running "


@app.post("/chat")
async def chat(request: ChatRequest):

    owner, repo = parse_github_repo(request.repository_url)

    tool_store = build_tool_store()

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
