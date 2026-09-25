import json
import os

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from qdrant_client.models import Distance, VectorParams

from src.qdrant_setup import client
from src.embedding_model import embedding_model

from rank_bm25 import BM25Okapi

bm25_indexes = {}
bm25_documents = {}


IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "myvenv" ".next",
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


def get_language(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".sql": "sql",
    }
    return language_map.get(extension, "unknown")


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


async def get_file_contents(
    session,
    owner,
    repo,
    path="",
):
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

    filename = os.path.basename(path)
    extension = os.path.splitext(filename)[1].lower()
    language = get_language(path)
    document = Document(
        page_content=file_content,
        metadata={
            "owner": owner,
            "repo": repo,
            "path": path,
            "source": (f"github:{owner}/{repo}/{path}"),
            "filename": filename,
            "extension": extension,
            "language": language,
        },
    )

    return [document]


async def fetch_repository(
    session,
    owner,
    repo,
    path="",
):
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

    language_map = {
        "python": Language.PYTHON,
        "javascript": Language.JS,
        "typescript": Language.TS,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "c": Language.CPP,
        "go": Language.GO,
        "rust": Language.RUST,
        "ruby": Language.RUBY,
        "php": Language.PHP,
        "html": Language.HTML,
        "css": Language.HTML,
        "markdown": Language.MARKDOWN,
    }

    all_chunks = []

    for document in documents:

        language = document.metadata.get("language", "unknown")

        if language in language_map:

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_map[language],
                chunk_size=1200,
                chunk_overlap=200,
            )

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
            )

        chunks = splitter.split_documents([document])
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata.get('path','unknown')}_{index}"
            )

        all_chunks.extend(chunks)

    print(f"Chunks created: {len(all_chunks)}")

    if all_chunks:
        print("\n--- SAMPLE CHUNK ---")
        print(all_chunks[0].page_content[:500])

        print("\n--- SAMPLE METADATA ---")
        print(all_chunks[0].metadata)

    return all_chunks


def create_bm25_index(chunks, collection_name):
    """
    Create a BM25 index from repository chunks.
    """
    tokenized_documents = [chunk.page_content.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_documents)
    bm25_indexes[collection_name] = bm25
    bm25_documents[collection_name] = chunks

    print(f"BM25 index created for: {collection_name}")
    print(f"BM25 documents: {len(chunks)}")

    return bm25


def bm25_search(query, collection_name, k=10):
    """
    Search repository chunks using BM25
    """
    if collection_name not in bm25_indexes:
        raise ValueError(f"BM25 index not found for {collection_name}")

    bm25 = bm25_indexes[collection_name]
    Documents = bm25_documents[collection_name]

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = []

    for index in ranked_indexes:
        document = Documents[index]

        results.append(
            {
                "document": document,
                "score": float(scores[index]),
            }
        )

    return results


def hybrid_search(query, collection_name, vectorstore, k=10):

    qdrant_result = vectorstore.similarity_search_with_score(
        query,
        k=30,
    )

    bm25_result = bm25_search(
        query,
        collection_name,
        k=30,
    )

    combined = {}

   
    for document, score in qdrant_result:

        chunk_id = document.metadata.get("chunk_id")

        if chunk_id not in combined:
            combined[chunk_id] = {
                "document": document,
                "qdrant_score": None,
                "bm25_score": None,
                "sources": [],
            }

        combined[chunk_id]["qdrant_score"] = float(score)
        combined[chunk_id]["sources"].append("qdrant")


    for result in bm25_result:

        document = result["document"]
        score = result["score"]

        chunk_id = document.metadata.get("chunk_id")

        if chunk_id not in combined:
            combined[chunk_id] = {
                "document": document,
                "qdrant_score": None,
                "bm25_score": None,
                "sources": [],
            }

        combined[chunk_id]["bm25_score"] = float(score)
        combined[chunk_id]["sources"].append("bm25")

    return list(combined.values())


def normalize_scores(results, score_key):

    scores = [
        result[score_key]
        for result in results
        if result[score_key] is not None
    ]

    if not scores:
        return

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for result in results:
            if result[score_key] is not None:
                result[f"{score_key}_normalized"] = 1.0
        return

    for result in results:

        score = result[score_key]

        if score is None:
            result[f"{score_key}_normalized"] = 0.0

        else:
            result[f"{score_key}_normalized"] = (
                (score - min_score)
                / (max_score - min_score)
            )


def rank_hybrid_results(results, top_k=15):

    normalize_scores(results, "qdrant_score")
    normalize_scores(results, "bm25_score")

    for result in results:

        qdrant_score = result.get(
            "qdrant_score_normalized",
            0.0
        )

        bm25_score = result.get(
            "bm25_score_normalized",
            0.0
        )

        result["hybrid_score"] = (
            0.5 * qdrant_score
            + 0.5 * bm25_score
        )

    results.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


def create_repo_store(
    chunks,
    owner,
    repo,
):
    collection_name = f"github_{owner}_{repo}".replace("/", "_")

    print(f"\nQdrant collection: " f"{collection_name}")

    if not client.collection_exists(collection_name):
        test_embedding = embedding_model.embed_query("test")

        vector_size = len(test_embedding)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print("Qdrant collection created.")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks to Qdrant.")

    else:
        print("Qdrant collection already exists.")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        print("Using existing repository vector store.")

    return vectorstore


async def build_repo_store(
    session,
    owner,
    repo,
):
    collection_name = f"github_{owner}_{repo}".replace("/", "_")

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

    create_bm25_index(
        chunks,
        collection_name,
    )

    return vectorstore


if __name__ == "__main__":
    print("repo_search.py loaded successfully")
