# GitHub Repository RAG Agent

An AI-powered GitHub repository assistant that combines **Agentic RAG, GitHub MCP, Qdrant, LangGraph, and Groq LLM** to understand and interact with GitHub repositories.

## Features

- Fetches GitHub repositories using GitHub MCP
- Extracts and filters source-code files
- Splits repository code into chunks
- Generates embeddings and stores them in Qdrant
- Performs semantic similarity search
- Dynamically selects GitHub MCP tools based on user queries
- Combines retrieved code with live GitHub data
- Generates grounded answers using Groq LLM
- Exposes the agent through FastAPI
- Supports LangGraph-based agent workflow

## Architecture

```text
User
 |
 v
FastAPI
 |
 v
LangGraph Agent
 |
 +-------------------+
 |                   |
 v                   v
Repository RAG     GitHub MCP
 |                   |
 v                   v
Qdrant Search      MCP Tools
 |                   |
 +---------+---------+
           |
           v
      Context Builder
           |
           v
        Groq LLM
           |
           v
        Response
RAG Pipeline
GitHub Repository
       |
       v
   GitHub MCP
       |
       v
   File Extraction
       |
       v
     Chunking
       |
       v
    Embeddings
       |
       v
     Qdrant
       |
       v
Similarity Search
       |
       v
Relevant Code
Tech Stack
Python
LangChain
LangGraph
GitHub MCP
Qdrant
Groq
FastAPI
Embedding Models
Example Queries
Explain this repository.
How does authentication work?
Where is JWT implemented?
Explain the RAG pipeline.
Show me the open issues.
Show me recent pull requests.
Running

Install dependencies:

pip install -r requirements.txt

Run the API:

uvicorn src.app:app --reload

API documentation:

http://127.0.0.1:8000/docs
Future Improvements
Hybrid search
Reranking
Code-aware chunking
Retrieval evaluation
Streaming responses
Conversation memory
Advanced repository understanding

**This is much better for GitHub:** short, readable, shows the architecture a