
import json
import math
import os
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from qdrant_client.models import Distance, VectorParams

from src.qdrant_setup import client
from src.embedding_model import embedding_model
from src.logger import logger

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def sigmoid(x: float) -> float:
    """Safely apply sigmoid to scale raw logits into (0, 1)."""
    try:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except OverflowError:
        return 1.0 if x > 0 else 0.0


bm25_indexes = {}
bm25_documents = {}

BM25_CACHE_DIR = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "bm25_cache"
)


def save_bm25_index(chunks, collection_name: str) -> None:
    """Save the BM25 index and document chunks to disk cache."""
    try:
        BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = BM25_CACHE_DIR / f"{collection_name}.pkl"
        bm25 = bm25_indexes.get(collection_name)
        if bm25 is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "bm25": bm25,
                        "chunks": chunks,
                    },
                    f,
                )
            logger.info("Saved BM25 index to disk cache: %s", cache_file)
    except Exception as e:
        logger.warning("Failed to save BM25 index to disk: %s", e)


def load_bm25_index(collection_name: str) -> bool:
    """Load the BM25 index and document chunks from disk cache if present."""
    if collection_name in bm25_indexes:
        return True

    cache_file = BM25_CACHE_DIR / f"{collection_name}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                bm25_indexes[collection_name] = data["bm25"]
                bm25_documents[collection_name] = data["chunks"]
            logger.info("Loaded BM25 index from disk cache for: %s", collection_name)
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 index from disk: %s", e)
            return False
    return False


# ============================================================
# IGNORED DIRECTORIES
# ============================================================

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "myvenv",
    ".next",
    "dist",
    "build",
    "coverage",
}


# ============================================================
# IGNORED FILES
# ============================================================

IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}


# ============================================================
# ALLOWED EXTENSIONS
# ============================================================

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


# ============================================================
# LANGUAGE DETECTION
# ============================================================

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


# ============================================================
# FILE FILTER
# ============================================================

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


# ============================================================
# EXTRACT FILE CONTENT
# ============================================================

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


# ============================================================
# EXTRACT DIRECTORY RESULT
# ============================================================

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


# ============================================================
# GET FILE CONTENTS
# ============================================================

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


# ============================================================
# FETCH SINGLE FILE
# ============================================================

async def fetch_file(
    session,
    owner,
    repo,
    path,
):

    logger.info("Reading file: %s", path)

    result = await get_file_contents(
        session,
        owner,
        repo,
        path,
    )

    file_content = extract_file_content(result)

    if not file_content:

        logger.warning("Could not extract content: %s", path)

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
            "source": f"github:{owner}/{repo}/{path}",
            "filename": filename,
            "extension": extension,
            "language": language,
        },
    )

    return [document]


# ============================================================
# FETCH ENTIRE REPOSITORY
# ============================================================

async def fetch_repository(
    session,
    owner,
    repo,
    path="",
):

    logger.info("Fetching repository path: %s", path or "/")

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

            # -------------------------
            # DIRECTORY
            # -------------------------

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

            # -------------------------
            # FILE
            # -------------------------

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


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    logger.info("Files loaded: %d", len(documents))

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

        language = document.metadata.get(
            "language",
            "unknown",
        )

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

        chunks = splitter.split_documents(
            [document]
        )

        path = document.metadata.get("path", "unknown")

        for index, chunk in enumerate(chunks):

            chunk_id = f"{path}_{index}"
            chunk.metadata["chunk_id"] = chunk_id

            # Contextual prefix so that embeddings, BM25, and Cross-Encoder
            # have full visibility into the file path and module name
            header = f"File: {path}\nLanguage: {language}\n\n"
            if not chunk.page_content.startswith("File: "):
                chunk.page_content = f"{header}{chunk.page_content}"

        all_chunks.extend(chunks)

    logger.info("Chunks created: %d", len(all_chunks))

    if all_chunks:
        logger.debug("Sample chunk: %s", all_chunks[0].page_content[:500])
        logger.debug("Sample metadata: %s", all_chunks[0].metadata)

    return all_chunks


# ============================================================
# CREATE BM25 INDEX
# ============================================================

def create_bm25_index(
    chunks,
    collection_name,
):

    """
    Create a BM25 index from repository chunks.
    """

    tokenized_documents = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    bm25_indexes[collection_name] = bm25

    bm25_documents[collection_name] = chunks

    logger.info("BM25 index created for: %s", collection_name)
    logger.info("BM25 documents: %d", len(chunks))

    save_bm25_index(chunks, collection_name)

    return bm25


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    collection_name,
    k=10,
):

    """
    Search repository chunks using BM25.
    """

    if collection_name not in bm25_indexes:

        # Attempt to load from disk cache first
        if not load_bm25_index(collection_name):
            raise ValueError(
                f"BM25 index not found for {collection_name}"
            )

    bm25 = bm25_indexes[collection_name]

    documents = bm25_documents[
        collection_name
    ]

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = []

    for index in ranked_indexes:

        document = documents[index]

        results.append(
            {
                "document": document,
                "score": float(
                    scores[index]
                ),
            }
        )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    collection_name,
    vectorstore,
    k=10,
):

    # --------------------------------
    # QDRANT SEARCH
    # --------------------------------

    qdrant_result = (
        vectorstore.similarity_search_with_score(
            query,
            k=30,
        )
    )

    # --------------------------------
    # BM25 SEARCH
    # --------------------------------

    bm25_result = bm25_search(
        query,
        collection_name,
        k=30,
    )

    # --------------------------------
    # MERGE RESULTS
    # --------------------------------

    combined = {}

    # --------------------------------
    # QDRANT RESULTS
    # --------------------------------

    for document, score in qdrant_result:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id not in combined:

            combined[chunk_id] = {
                "document": document,
                "qdrant_score": None,
                "bm25_score": None,
                "sources": [],
            }

        combined[chunk_id][
            "qdrant_score"
        ] = float(score)

        combined[chunk_id][
            "sources"
        ].append("qdrant")

    # --------------------------------
    # BM25 RESULTS
    # --------------------------------

    for result in bm25_result:

        document = result["document"]

        score = result["score"]

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id not in combined:

            combined[chunk_id] = {
                "document": document,
                "qdrant_score": None,
                "bm25_score": None,
                "sources": [],
            }

        combined[chunk_id][
            "bm25_score"
        ] = float(score)

        combined[chunk_id][
            "sources"
        ].append("bm25")
        logger.debug(
            "Sample metadata: %s",
            combined[next(iter(combined))]["document"].metadata,
        )

    return list(
        combined.values()
    )


# ============================================================
# NORMALIZE SCORES
# ============================================================

def normalize_scores(
    results,
    score_key,
):

    scores = [
        result[score_key]
        for result in results
        if result[score_key] is not None
    ]

    if not scores:
        return

    min_score = min(scores)

    max_score = max(scores)

    # --------------------------------
    # ALL SCORES ARE THE SAME
    # --------------------------------

    if max_score == min_score:

        for result in results:

            if result[score_key] is not None:

                result[
                    f"{score_key}_normalized"
                ] = 1.0

        return

    # --------------------------------
    # MIN-MAX NORMALIZATION
    # --------------------------------

    for result in results:

        score = result[score_key]

        if score is None:

            result[
                f"{score_key}_normalized"
            ] = 0.0

        else:

            result[
                f"{score_key}_normalized"
            ] = (
                (score - min_score)
                / (max_score - min_score)
            )


# ============================================================
# HYBRID RANKING
# ============================================================

def rank_hybrid_results(results, query, top_k=15):

    # 1. Normalize Qdrant and BM25 scores
    normalize_scores(results, "qdrant_score")
    normalize_scores(results, "bm25_score")

    # 2. Calculate hybrid score
    for result in results:

        qdrant_score = result.get(
            "qdrant_score_normalized", 0.0
        )

        bm25_score = result.get(
            "bm25_score_normalized", 0.0
        )

        result["hybrid_score"] = (
            0.5 * qdrant_score
            + 0.5 * bm25_score
        )

    # 3. First rank using hybrid retrieval
    results.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    # 4. Send query + document to Cross-Encoder
    pairs = []

    logger.debug("Reranker input document count: %d", len(results))

    for result in results:

        document = result["document"]
        metadata = document.metadata

        logger.debug(
            "Reranker input: file=%s chunk=%s content=%s",
            metadata.get("path"),
            metadata.get("chunk_id"),
            document.page_content[:1000],
        )

        pairs.append(
            (
                query,
                document.page_content
            )
        )

    # 5. Get reranker scores
    reranker_scores = reranker.predict(pairs)

    # 6. Store reranker scores safely
    for result, score in zip(
        results,
        reranker_scores
    ):
        raw_score = float(score)
        # BGE CrossEncoder can return unbounded logits depending on version
        if raw_score < -1.0 or raw_score > 1.0:
            raw_score = sigmoid(raw_score)
        result["reranker_score"] = raw_score

    # 7. Final ranking using Cross-Encoder
    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    for result in results[:top_k]:
        metadata = result["document"].metadata
        logger.info(
            "Reranked | file=%s | chunk=%s | score=%.6f",
            metadata.get("path", "unknown"),
            metadata.get("chunk_id", "unknown"),
            result["reranker_score"],
        )

    # 8. Return final top results
    return results[:top_k]


# ============================================================
# CREATE REPOSITORY VECTOR STORE
# ============================================================

def create_repo_store(
    chunks,
    owner,
    repo,
):

    collection_name = (
        f"github_{owner}_{repo}"
        .replace("/", "_")
    )

    logger.info("Qdrant collection: %s", collection_name)

    # --------------------------------
    # CREATE COLLECTION
    # --------------------------------

    if not client.collection_exists(
        collection_name
    ):

        test_embedding = (
            embedding_model.embed_query(
                "test"
            )
        )

        vector_size = len(
            test_embedding
        )

        client.create_collection(
            collection_name=collection_name,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        logger.info("Qdrant collection created.")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        vectorstore.add_documents(
            chunks
        )

        logger.info("Added %d chunks to Qdrant.", len(chunks))

    # --------------------------------
    # EXISTING COLLECTION
    # --------------------------------

    else:

        logger.info("Qdrant collection already exists.")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        logger.info("Using existing repository vector store.")

    return vectorstore


# ============================================================
# BUILD REPOSITORY STORE
# ============================================================

async def build_repo_store(
    session,
    owner,
    repo,
    force_refresh=False,
):

    collection_name = (
        f"github_{owner}_{repo}"
        .replace("/", "_")
    )

    # Check if both Qdrant collection and BM25 index are already cached
    if not force_refresh:
        collection_exists = client.collection_exists(collection_name)
        bm25_cached = load_bm25_index(collection_name)

        if collection_exists and bm25_cached:
            logger.info(
                "Reusing existing Qdrant collection and cached BM25 index for: %s (skipping GitHub crawl)",
                collection_name,
            )
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_model,
            )

    logger.info("Ingesting repository from GitHub: %s/%s", owner, repo)

    documents = await fetch_repository(
        session,
        owner,
        repo,
    )

    if not documents:

        raise ValueError(
            "No repository files were found."
        )

    chunks = split_documents(
        documents
    )

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


def filter_relevant_evidence(
    results,
    top_k=5,
    min_k=3,
    relative_threshold=0.2,
):
    """
    Filter evidence chunks adaptively while ensuring that:
    1. A minimum floor of evidence chunks (min_k) is preserved.
    2. Real source code files are not starved out by README/documentation.
    3. Low-scoring noise is pruned.
    """
    if not results:
        return []

    # If results are already fewer than or equal to min_k, keep all of them up to top_k
    if len(results) <= min_k:
        return results[:top_k]

    scores = [
        float(result.get("reranker_score", 0.0))
        for result in results
    ]

    max_score = max(scores)

    # Dynamic cutoff
    if max_score > 0:
        cutoff = max_score * relative_threshold
        filtered_results = [
            result
            for result in results
            if float(result.get("reranker_score", 0.0)) >= cutoff
        ]
    else:
        filtered_results = list(results)

    # Safety floor: ensure at least min_k items are kept
    if len(filtered_results) < min_k:
        filtered_results = results[:min_k]

    # Code-preservation heuristic:
    # If the filtered results only contain documentation (e.g. .md, .txt),
    # but top reranked results contain source code (.js, .py, .ts, etc.),
    # include the best code chunk(s) so the LLM has actual implementation evidence!
    has_code = any(
        res["document"].metadata.get("extension", "") not in {".md", ".txt", ""}
        for res in filtered_results
    )
    if not has_code:
        for res in results:
            ext = res["document"].metadata.get("extension", "")
            if ext not in {".md", ".txt", ""} and res not in filtered_results:
                filtered_results.append(res)
                break

    return filtered_results[:top_k]


def remove_redundant_evidence(
    results,
    embeddings,
    similarity_threshold=0.85
):
    """
    Remove chunks that are semantically too similar
    to an already selected chunk.

    The results should already be ordered by reranker score.
    """

    if not results:
        logger.info("Redundancy filter received 0 results.")
        return []

    selected = []
    selected_embeddings = []

    logger.info("========== REDUNDANCY FILTER ==========")
    logger.info(
        "Redundancy filter input count: %d | threshold=%.6f",
        len(results),
        similarity_threshold,
    )

    for result in results:

        document = result["document"]
        metadata = document.metadata
        text = document.page_content

        # Create embedding for the current chunk
        embedding = embeddings.embed_query(text)

        # First document is always selected
        if not selected_embeddings:

            selected.append(result)
            selected_embeddings.append(embedding)

            logger.info(
                "Redundancy check | file=%s | chunk=%s | "
                "similarity=N/A | threshold=%.4f | KEEP",
                metadata.get("path", "unknown"),
                metadata.get("chunk_id", "unknown"),
                similarity_threshold,
            )

            continue

        # Compare current chunk with all previously selected chunks
        similarities = cosine_similarity(
            [embedding],
            selected_embeddings
        )[0]

        max_similarity = float(max(similarities))

        # Keep only if it is sufficiently different
        if max_similarity < similarity_threshold:

            selected.append(result)
            selected_embeddings.append(embedding)

            decision = "KEEP"

        else:

            decision = "REMOVE"

        logger.info(
            "Redundancy check | file=%s | chunk=%s | "
            "reranker_score=%.6f | similarity=%.4f | "
            "threshold=%.4f | %s",
            metadata.get("path", "unknown"),
            metadata.get("chunk_id", "unknown"),
            float(result.get("reranker_score", 0.0)),
            max_similarity,
            similarity_threshold,
            decision,
        )

    logger.info(
        "Redundancy filtering: %d -> %d",
        len(results),
        len(selected),
    )

    logger.info("=======================================")

    return selected
    
    
def select_complementary_evidence(results, max_chunks=8):
    """
    Select evidence that provides different information.

    Results are already:
    1. reranked
    2. relevance filtered
    3. redundancy filtered
    """

    selected = []
    selected_files = set()

    for result in results:
        document = result["document"]
        metadata = document.metadata

        path = metadata.get("path", "")

        # Prefer evidence from different files
        if path not in selected_files:
            selected.append(result)
            selected_files.add(path)

        if len(selected) >= max_chunks:
            break

    return selected


def select_metadata_aware_evidence(results, max_chunks=8):
    """
    Select evidence while maintaining diversity across
    files and source types.
    """

    selected = []
    selected_paths = set()
    selected_extensions = set()

    # First pass:
    # Prefer strong results from different files.
    for result in results:
        document = result["document"]
        metadata = document.metadata

        path = metadata.get("path", "")
        extension = metadata.get("extension", "")

        if path in selected_paths:
            continue

        selected.append(result)
        selected_paths.add(path)
        selected_extensions.add(extension)

        if len(selected) >= max_chunks:
            return selected

    return selected


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logger.info("repo_search.py loaded successfully")
