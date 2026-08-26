# import asyncio
# from mcp_setup import mcp_session


# async def main():

#     async with mcp_session() as session:

#         result = await session.list_tools()

#         for tool in result.tools:

#             if tool.name == "issue_read":

#                 print("\nTOOL:")
#                 print(tool.name)

#                 print("\nDESCRIPTION:")
#                 print(tool.description)

#                 print("\nINPUT SCHEMA:")
#                 print(tool.inputSchema)


# if __name__ == "__main__":
#     asyncio.run(main())

from qdrant_setup import client
from embedding_model import embedding_model
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


documents = [
    Document(
        page_content="This is a test document for Qdrant."
    )
]

print("Creating test collection...")

vectorstore = QdrantVectorStore.from_documents(
    documents,
    embedding_model,
    client=client,
    collection_name="test_collection",
)

print("Qdrant vector store created successfully!")