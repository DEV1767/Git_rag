from typing import Any, Dict, List

from src.repo_search import (
    hybrid_search,
    rank_hybrid_results,
    filter_relevant_evidence,
    remove_redundant_evidence,
)

from src.embedding_model import embedding_model
from src.logger import logger


def retrieve_hybrid_documents(
    vectorstore: Any,
    query: str,
    collection_name: str,
    k: int = 30,
) -> List[Dict[str, Any]]:
    """
    Retrieves, ranks, filters, and deduplicates documents
    using a hybrid search approach.

    Args:
        vectorstore: The vector database instance.
        query (str): The search query.
        collection_name (str): The name of the collection to search.
        k (int): Initial number of documents to retrieve.

    Returns:
        List[Dict[str, Any]]: Processed evidence results.
    """

    # ---------------------------------------------------------
    # Step 1: Hybrid Search
    # ---------------------------------------------------------

    results = hybrid_search(
        query=query,
        collection_name=collection_name,
        vectorstore=vectorstore,
        k=k,
    )

    logger.info(
        "AFTER HYBRID SEARCH: %d",
        len(results),
    )

    # ---------------------------------------------------------
    # Step 2: Hybrid + Cross-Encoder Ranking
    # ---------------------------------------------------------

    results = rank_hybrid_results(
        results,
        query=query,
        top_k=15,
    )

    logger.info(
        "AFTER RERANKING: %d",
        len(results),
    )

    # ---------------------------------------------------------
    # Step 3: Adaptive Relevance Filtering
    # ---------------------------------------------------------

    results = filter_relevant_evidence(
        results,
        top_k=5,
        relative_threshold=0.2,
    )

    logger.info(
        "AFTER EVIDENCE FILTER: %d",
        len(results),
    )

    # ---------------------------------------------------------
    # Step 4: Remove Redundant Evidence
    # ---------------------------------------------------------

    results = remove_redundant_evidence(
        results,
        embeddings=embedding_model,
        similarity_threshold=0.85,
    )

    logger.info(
        "AFTER REDUNDANCY FILTER: %d",
        len(results),
    )

    # ---------------------------------------------------------
    # Step 5: Final Evidence
    # ---------------------------------------------------------

    logger.info(
        "FINAL EVIDENCE COUNT: %d",
        len(results),
    )

    return results


def build_context(
    results: List[Dict[str, Any]]
) -> str:
    """
    Convert selected evidence chunks into a structured
    context string for the LLM.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        path = metadata.get("path", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")
        language = metadata.get("language", "unknown")
        score = float(result.get("reranker_score", 0.0))

        content = document.page_content

        context_parts.append(
            f"========== EVIDENCE {i} ==========\n"
            f"File: {path}\n"
            f"Chunk: {chunk_id}\n"
            f"Language: {language}\n"
            f"Reranker Score: {score:.6f}\n"
            f"Source Type: Repository Source Code / Documentation\n"
            f"\n"
            f"{content}\n"
            f"====================================\n"
        )

    return "\n".join(context_parts)