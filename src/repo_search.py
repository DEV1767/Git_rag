import json
import os

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_setup import client
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client.models import Distance, VectorParams
from embedding_model import embedding_model


IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".next",
    "dist",
    "build",
    "coverage",
}

IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".sql",
}


def should_include_file(path: str) -> bool:

    filename = os.path.basename(path)

    if filename in IGNORED_FILES:
        return False

    parts = path.split("/")

    for part in parts:
        if part in IGNORED_DIRECTORIES:
            return False

    extension = os.path.splitext(filename)[1].lower()

    if extension in ALLOWED_EXTENSIONS:
        return True

    if filename in {
        "README",
        "README.md",
        "Dockerfile",
        ".gitignore",
    }:
        return True

    return False


def extract_file_content(tool_result):

    for content in tool_result.content:

        if hasattr(content, "resource"):

            resource = content.resource

            if hasattr(resource, "text") and resource.text:
                return resource.text

        if hasattr(content, "text") and content.text:

            text = content.text

            try:
                data = json.loads(text)

                if isinstance(data, dict):
                    if data.get("content"):
                        return data["content"]

            except json.JSONDecodeError:
                pass

            if not text.startswith("successfully downloaded"):
                return text

    return None


def extract_directory_result(tool_result):

    output = []

    for content in tool_result.content:

        if hasattr(content, "text") and content.text:

            try:
                data = json.loads(content.text)
                return data
            except json.JSONDecodeError:
                continue

    return output


async def get_file_contents(session, owner, repo, path=""):

    return await session.call_tool(
        "get_file_contents",
        {
            "owner": owner,
            "repo": repo,
            "path": path,
        },
    )


async def fetch_file(
    session,
    owner,
    repo,
    path,
):

    print(f"Reading: {path}")

    result = await get_file_contents(
        session,
        owner,
        repo,
        path,
    )

    file_content = extract_file_content(result)

    if not file_content:

        print(f"Could not extract content: {path}")

        return []

    document = Document(
        page_content=file_content,
        metadata={
            "owner": owner,
            "repo": repo,
            "path": path,
            "source": f"github:{owner}/{repo}/{path}",
        },
    )

    return [document]


async def fetch_repository(session, owner, repo, path=""):

    print(f"Fetching: {path or '/'}")

    result = await get_file_contents(
        session,
        owner,
        repo,
        path,
    )

    data = extract_directory_result(result)

    documents = []

    if isinstance(data, list):

        for item in data:

            item_path = item.get("path")
            item_type = item.get("type")

            if not item_path:
                continue

            if item_type == "dir":

                directory_name = os.path.basename(item_path)

                if directory_name in IGNORED_DIRECTORIES:
                    continue

                child_documents = await fetch_repository(
                    session,
                    owner,
                    repo,
                    item_path,
                )

                documents.extend(child_documents)

            elif item_type == "file":

                if not should_include_file(item_path):
                    continue

                file_documents = await fetch_file(
                    session,
                    owner,
                    repo,
                    item_path,
                )

                documents.extend(file_documents)

    elif isinstance(data, dict):

        if data.get("type") == "file":

            if should_include_file(path):

                documents = await fetch_file(
                    session,
                    owner,
                    repo,
                    path,
                )

    return documents


def split_documents(documents):

    print(f"\nFiles loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    return chunks


def create_repo_store(
    chunks,
    owner,
    repo,
):
    collection_name = f"github_{owner}_{repo}".replace("/", "_")

    print(f"\nCreating Qdrant collection: " f"{collection_name}")
    test_embedding = embedding_model.embed_query("test")

    vector_size = len(test_embedding)


    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print("Qdrant collection created.")

    else:
        print("Qdrant collection already exists.")

 
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    vectorstore.add_documents(chunks)

    print("Repository vector store created.")

    return vectorstore


async def build_repo_store(
    session,
    owner,
    repo,
):

    documents = await fetch_repository(
        session,
        owner,
        repo,
    )

    if not documents:

        raise ValueError("No repository files were found.")

    chunks = split_documents(documents)

    vectorstore = create_repo_store(
        chunks,
        owner,
        repo,
    )

    return vectorstore


if __name__ == "__main__":
    print("repo_loader.py loaded successfully")
