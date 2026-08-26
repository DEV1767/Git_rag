import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from src.embedding_model import embedding_model


load_dotenv()


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def build_tool_store():

    collection_name = "mcp_tools"

   
    collections = client.get_collections()

    collection_exists = any(
        collection.name == collection_name
        for collection in collections.collections
    )

    
    if not collection_exists:

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{collection_name}"
        )

    else:

        print(
            f"Using existing Qdrant collection: "
            f"{collection_name}"
        )

    
    tool_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    return tool_store


print("DB connected")