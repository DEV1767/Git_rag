
from src.repo_search import hybrid_search, rank_hybrid_results
from langchain_core.documents import Document


def retrieve_hybrid_documents(
    vectorstore,
    query,
    collection_name,
    k=30,
):
    # Get candidates from Qdrant + BM25
    results = hybrid_search(
        query=query,
        collection_name=collection_name,
        vectorstore=vectorstore,
        k=k,
    )

    # Hybrid ranking
    # Rank the combined candidates and keep the top 8
    results = rank_hybrid_results(
        results,
        top_k=15,
    )

    print("\nFinal Hybrid Ranked Results:")

    for result in results:

        document = result["document"]

        print(
            "-",
            document.metadata.get("path"),
            "| chunk:",
            document.metadata.get("chunk_id"),
            "| qdrant:",
            result.get("qdrant_score"),
            "| bm25:",
            result.get("bm25_score"),
            "| hybrid:",
            round(
                result.get("hybrid_score", 0.0),
                4,
            ),
            "| sources:",
            result.get("sources"),
        )

    print(
        f"\nFinal hybrid chunks: {len(results)}"
    )

    return results


def build_context(results):

    context = []

    for result in results:

        document = result["document"]

        context.append(
            f"FILE: {document.metadata.get('path')}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context)

