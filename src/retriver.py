from langchain_core.documents import Document


def retrieve_documents(vectorstore, query, k=5):
    documents = vectorstore.similarity_search(query, k=k)

    print("\nRelevant repository files:")

    for doc in documents:
        print("-", doc.metadata.get("path"))

    print(f"\nRepository chunks retrieved: {len(documents)}")

    return documents


def build_context(documents):
    context = []

    for doc in documents:
        context.append(f"FILE: {doc.metadata.get('path')}\n" f"{doc.page_content}")

    return "\n\n".join(context)
