from langchain_huggingface import HuggingFaceEmbeddings

MODEL_PATH = r"C:\Users\Shivam\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5"

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)