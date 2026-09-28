from qdrant_setup import client
from embedding_model import embedding_model
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from logger import logger


documents = [
    Document(
        page_content="This is a test document for Qdrant."
    )
]

logger.info("Creating test collection...")

vectorstore = QdrantVectorStore.from_documents(
    documents,
    embedding_model,
    client=client,
    collection_name="test_collection",
)

logger.info("Qdrant vector store created successfully")