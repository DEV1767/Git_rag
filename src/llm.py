import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


OpenRouter_model=ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,


)


embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

google_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

Groq_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

