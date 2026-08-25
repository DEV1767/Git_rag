from langchain_huggingface import HuggingFaceEmbeddings

MODEL_PATH =  r"C:\Users\Shivam\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)