# GitHub Agentic RAG - Project Context

## 1. Purpose

This project is a GitHub repository assistant. A user provides a GitHub repository
URL (or an `owner/repository` identifier) and asks a natural-language question.
The application:

1. Connects to GitHub through the GitHub MCP server.
2. Reads and indexes relevant repository files in Qdrant.
3. Selects a live GitHub MCP tool when the question requires GitHub metadata or
   operations such as issues or pull requests.
4. Retrieves semantically relevant code chunks.
5. Combines repository context and live MCP output.
6. Uses a Groq-hosted chat model to generate a grounded answer.

The intended result is an explainable repository-aware assistant rather than a
general-purpose chatbot. The answer should be based on the selected repository
context and should mention useful file names.

## 2. User-facing behavior

The web UI is served from `src/static/index.html`. It is a dark, terminal-style
single-page interface with:

- A repository field accepting either `owner/repository` or a full GitHub URL.
- A question textarea.
- A submit button that sends a JSON request to `POST /chat`.
- Loading messages while the repository is read, tools are called, and the answer
  is generated.
- An answer panel showing the returned answer as plain text.
- A status line showing either the selected MCP tool or that the answer is ready.

Example questions:

- `Explain this repository.`
- `How does authentication work?`
- `Where is JWT implemented?`
- `Explain the RAG pipeline.`
- `Show me the open issues.`
- `Show me recent pull requests.`

## 3. High-level architecture

```text
Browser
  |
  | GET / and POST /chat
  v
FastAPI (src/app.py)
  |
  +--> Parse repository identifier
  |
  +--> Qdrant tool store
  |
  +--> Dockerized GitHub MCP session
  |      |
  |      +--> Discover/list or execute GitHub tools
  |      +--> Read repository files
  |
  +--> Repository vector store in Qdrant
  |
  v
LangGraph workflow (src/agent.py)
  |
  +--> Select relevant MCP tool
  |       |
  |       +--> Use cached semantic tool descriptions
  |       +--> Discover MCP tools if the cache has no relevant match
  |
  +--> Optional MCP execution
  |
  +--> Semantic repository retrieval
  |
  +--> Groq answer generation
  v
Response JSON -> Browser
```

The project has two related retrieval systems:

1. **Repository RAG store**: stores chunks of source files for semantic code
   retrieval.
2. **MCP tool store**: stores descriptions of GitHub MCP tools so the agent can
   choose a tool by semantic similarity before asking the LLM to make the final
   selection.

## 4. Request lifecycle

### 4.1 FastAPI request handling

`src/app.py` defines:

- `GET /`: returns the static HTML UI.
- `GET /health`: returns `{"status": "ok"}`.
- `POST /chat`: accepts a `ChatRequest` with:
  - `repository_url: str`
  - `question: str`

For `/chat`, the application:

1. Returns a friendly answer immediately if the question is empty.
2. Calls `parse_github_repo` to normalize the input into `owner` and `repo`.
3. Builds the global MCP tool vector store.
4. Opens a GitHub MCP session.
5. Builds or reuses the repository-specific vector store.
6. Builds and invokes the LangGraph agent with the question, repository identity,
   both stores, and the MCP session.
7. Returns:

```json
{
  "repository": "owner/repository",
  "question": "the original question",
  "selected_tool": "tool name or null",
  "answer": "generated answer"
}
```

Known HTTP error handling:

- Invalid repository input: HTTP 400.
- Qdrant unavailable: HTTP 503.
- Docker/GitHub MCP startup or initialization failure: HTTP 503.
- Qdrant failure while building repository context: HTTP 503.

## 5. LangGraph agent workflow

`src/agent.py` defines the `AgentState` TypedDict and compiles a graph with
these nodes:

```text
START
  -> select_tool
       -> execute_mcp -> retrieve_rag -> generate_answer -> END
       -> retrieve_rag -> generate_answer -> END
```

### `select_tool`

`select_tool_node` calls `src.tools.tool_selector.select_tool`. The selector
searches the MCP tool store for the five closest tool descriptions and asks the
Groq model to return exactly one tool name or `NO_RELEVANT_TOOL`.

Routing behavior:

- `NO_RELEVANT_TOOL` routes directly to repository retrieval.
- Any other returned value routes to MCP execution.

### `execute_mcp`

`execute_mcp_node` calls the selected MCP tool with:

```python
{
    "owner": state["owner"],
    "repo": state["repo"],
}
```

The raw MCP result is converted to a string and stored as `mcp_context`.

### `retrieve_rag`

`retrieve_rag_node` runs semantic similarity search against the repository
vector store with `k=15`. It converts the returned documents into context entries
formatted as:

```text
FILE: path/to/file
<file chunk contents>
```

If the MCP branch was not used, `mcp_context` is set to
`No MCP information was required.` The MCP branch still performs repository
retrieval, so live GitHub information and code context can be used together.

### `generate_answer`

The node combines:

- `Repository Code Context`
- `GitHub MCP Context`

It invokes `retriever_prompt` and sends the prompt to `Groq_model` asynchronously.
The model output is normalized into a string and returned as `answer`.

The answer prompt instructs the model to:

- Use only the supplied context.
- Explain the code clearly and concisely.
- Mention relevant file names.
- Avoid inventing information.
- Say `I could not find the answer in the repository.` when the answer is not
  available in the provided context.

## 6. GitHub MCP integration

`src/mcp_setup.py` starts the official GitHub MCP server as a Docker subprocess:

```text
docker run -i --rm
  -e GITHUB_PERSONAL_ACCESS_TOKEN
  ghcr.io/github/github-mcp-server
```

The MCP client:

1. Creates `StdioServerParameters`.
2. Connects to the Docker process over stdio.
3. Creates an `mcp.ClientSession`.
4. Calls `session.initialize()`.
5. Yields the session to the caller.

`execute_mcp_tool` is a thin wrapper around `session.call_tool`.

The MCP server is used for two different jobs:

- Enumerating and describing available GitHub tools.
- Reading repository files and executing the selected live GitHub tool.

## 7. Repository ingestion and indexing

`src/repo_search.py` implements the repository ingestion pipeline.

### Repository parsing

`src/repo_helper.py::parse_github_repo` accepts:

- A URL matching `https://github.com/<owner>/<repo>`.
- An `owner/repo` string.

It removes a trailing `.git` from URL repositories and raises `ValueError` for
unsupported input.

### Recursive file traversal

`fetch_repository` calls the GitHub MCP `get_file_contents` tool recursively.
Directories are traversed and files are fetched individually.

Included file extensions:

```text
.py .js .jsx .ts .tsx .java .cpp .c .h .hpp .go .rs .php .rb
.html .css .scss .json .yaml .yml .md .txt .sql
```

Also included by name:

- `README`
- `README.md`
- `Dockerfile`
- `.gitignore`

Ignored directories:

```text
.git, node_modules, __pycache__, .venv, venv, env,
.next, dist, build, coverage
```

Ignored lock files:

```text
package-lock.json, yarn.lock, pnpm-lock.yaml,
poetry.lock, Pipfile.lock
```

Each fetched file becomes a LangChain `Document` containing:

- File content in `page_content`.
- `owner`, `repo`, and `path` metadata.
- A `source` value in the form `github:owner/repo/path`.

The content extraction code supports MCP text/resource responses and JSON
responses containing a `content` field.

### Chunking

`split_documents` uses `RecursiveCharacterTextSplitter` with:

- `chunk_size=1200`
- `chunk_overlap=200`

This preserves some neighboring context between chunks while keeping embedding
inputs reasonably small.

### Repository collection

Each repository gets a Qdrant collection named:

```text
github_<owner>_<repo>
```

If the collection does not exist, the application:

1. Fetches repository files.
2. Splits them into chunks.
3. Creates a cosine-distance collection using the detected embedding dimension.
4. Inserts the chunks.

If the collection already exists, it is reused and the repository is not fetched
or re-indexed. There is currently no automatic refresh, commit/version tracking,
or invalidation strategy.

## 8. Embeddings and Qdrant

### Code embeddings

`src/embedding_model.py` creates a CPU-based
`HuggingFaceEmbeddings` instance using the local BAAI BGE-small English model.
Embeddings are normalized before storage/search. The configured model path is
currently an absolute Windows path:

```text
C:\Users\Shivam\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5\...
```

The Qdrant vector size for the tool collection is hard-coded as `384`. Repository
collections determine their size by embedding a test string, which is safer if
the model changes.

### Qdrant client

`src/qdrant_setup.py` loads environment variables and creates a Qdrant client
using:

- `QDRANT_URL`
- `QDRANT_API_KEY`

The shared MCP tool collection is named `mcp_tools`. It is created on demand
with cosine distance and the BGE-small dimension.

## 9. MCP tool discovery and selection

`src/tools/tool_selector.py` contains the tool-selection subsystem.

### Discovery

`discover_mcp_tools` calls `session.list_tools()` and converts each tool to a
LangChain `Document` containing:

- The tool description as page content.
- Tool name.
- Source `github_mcp`.
- Stringified input schema.

`add_mcp_to_store` adds these documents to the `mcp_tools` Qdrant collection.

### Selection

For every question:

1. Search the tool store for five semantically similar tool descriptions.
2. Render candidate names and descriptions into `tool_selection_prompt`.
3. Ask Groq to return exactly one candidate name or `NO_RELEVANT_TOOL`.
4. If no tool is selected, discover all current MCP tools, index them, search
   again, and ask Groq a second time.

The tool-selection prompt explicitly forbids invented tool names, explanations,
markdown, and direct question answering.

## 10. Models and prompts

`src/llm.py` loads `.env` with `python-dotenv` and defines:

- `Groq_model`: `ChatGroq(model="openai/gpt-oss-120b", temperature=0)`.
- `google_model`: `ChatGoogleGenerativeAI(model="gemini-2.5-flash",
  temperature=0)`.
- `embedding`: `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`.

The active agent uses `Groq_model` for both tool selection and final answer
generation. The Google model and Google embedding object are currently defined
but not used by the main FastAPI/LangGraph path.

Prompt definitions live in `src/prompt_helper.py`:

- `tool_selection_prompt`: strict single-tool selection.
- `retriever_prompt`: grounded repository assistant answer generation.

## 11. Important files

| File | Responsibility |
|---|---|
| `src/app.py` | FastAPI application, HTTP routes, request validation, error mapping |
| `src/agent.py` | LangGraph state, nodes, routing, CLI entry point |
| `src/mcp_setup.py` | Dockerized GitHub MCP session and tool execution |
| `src/repo_helper.py` | Repository identifier parsing |
| `src/repo_search.py` | GitHub traversal, file filtering, extraction, chunking, repository indexing |
| `src/retriver.py` | Similarity retrieval and context formatting |
| `src/qdrant_setup.py` | Qdrant client and MCP tool vector store |
| `src/embedding_model.py` | Local Hugging Face embedding model |
| `src/tools/tool_selector.py` | MCP tool discovery, indexing, and LLM-assisted selection |
| `src/prompt_helper.py` | Tool-selection and answer-generation prompts |
| `src/llm.py` | Groq and Google model construction |
| `src/static/index.html` | Browser UI and `/chat` client |
| `main.py` | Convenience Uvicorn launcher |
| `src/re_ranker.py` | Standalone CrossEncoder reranker prototype; not wired into the active pipeline |
| `src/test.py` | Manual Qdrant test script; not a conventional automated test suite |
| `tool.txt` | Captured/example list of GitHub MCP tools available during development |
| `requirement.txt` | Python dependencies |
| `.env` | Local secrets/configuration; intentionally not documented with values |
| `.gitignore` | Ignores `.env`, the local virtual environment, and logs |

## 12. Runtime configuration

The application expects a local `.env` containing credentials/configuration for
the services used by the code. The important variable names are:

```text
GITHUB_PERSONAL_ACCESS_TOKEN
QDRANT_URL
QDRANT_API_KEY
GROQ_API_KEY
```

Do not commit the actual values. Docker Desktop must be running because the GitHub
MCP server is started through Docker. The local Hugging Face embedding model must
also exist at the configured absolute path, or `src/embedding_model.py` must be
changed to a portable model reference/path.

## 13. Installation and startup

Install dependencies:

```powershell
pip install -r requirement.txt
```

Start the development server from the repository root:

```powershell
uvicorn src.app:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Swagger/OpenAPI docs: `http://127.0.0.1:8000/docs`

## 14. Current limitations and implementation notes

These are important when modifying the project:

1. **Repository collections are cached forever by name.** A changed GitHub
   repository will continue using stale vectors until the collection is deleted
   or a refresh strategy is implemented.
2. **The cache key does not include a branch or commit.** The indexed content is
   not tied to a specific repository revision.
3. **The embedding model path is machine-specific.** A different Windows user,
   machine, Linux environment, or deployment container will not have the same
   absolute path by default.
4. **The tool collection is global.** Tool documents from repeated MCP discovery
   calls may be inserted repeatedly unless Qdrant deduplicates them.
5. **MCP execution arguments are generic.** The active execution node passes only
   `owner` and `repo`, even though different GitHub MCP tools may require
   additional arguments. Tool selection therefore works best for tools whose
   schema can be satisfied by those two values.
6. **The final answer prompt receives the stringified raw MCP result.** There is
   no structured normalization of issue, pull-request, or file results.
7. **The reranker is unused.** `src/re_ranker.py` defines a BGE CrossEncoder but
   `retrieve_documents` currently returns Qdrant similarity results directly.
8. **There is no automated test suite.** `src/test.py` is an exploratory/manual
   Qdrant script rather than unit or integration tests.
9. **The main `/chat` request builds/reuses stores per request.** Existing
   Qdrant collections avoid re-ingestion, but the client and graph setup are not
   implemented as an explicit application lifespan cache.
10. **The project is currently synchronous in its embedding/Qdrant retrieval
    operations inside an async graph.** The network-facing orchestration is async,
    while several vector-store calls are regular synchronous calls.
11. **Prompt grounding is instruction-based.** The final model is told not to
    hallucinate, but there is no citation validator, answer verifier, retrieval
    score threshold, or evaluation harness.
12. **Unused imports/prototypes exist.** Some modules contain development-era
    imports or experimental code; preserve behavior before cleaning them up.

## 15. Safe extension points

When adding features, prefer these boundaries:

- Add request/response behavior in `src/app.py`.
- Add workflow stages or routing in `src/agent.py`.
- Add GitHub access and response parsing in `src/mcp_setup.py` or
  `src/repo_search.py`.
- Add retrieval improvements in `src/retriver.py` and `src/repo_search.py`.
- Add tool-selection behavior in `src/tools/tool_selector.py`.
- Keep prompt changes centralized in `src/prompt_helper.py`.
- Keep model/provider construction centralized in `src/llm.py`.
- Keep Qdrant collection/client behavior centralized in `src/qdrant_setup.py`.

Any future change should preserve the core invariant: the final response must be
grounded in the requested repository's retrieved code and/or explicitly returned
GitHub MCP data, rather than unsupported model knowledge.
