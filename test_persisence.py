import streamlit as st
import chromadb
from chromadb.config import Settings
from vectorizer import vectorize
from chromadb.utils import embedding_functions
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from dotenv import dotenv_values
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import VectorDBQA
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

env_openai = dotenv_values(".env")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = env_openai["OPENAI_API_KEY"],
    model_name = "text-embedding-ada-002"
)

llm = OpenAI(temperature=0, model="text-embedding-ada-002", openai_api_key = env_openai["OPENAI_API_KEY"])

# Access collection
chroma_db = chromadb.Client()
collection = chroma_db.get_collection(name="rfp_generator", embedding_function=openai_ef)

print(collection.peek())