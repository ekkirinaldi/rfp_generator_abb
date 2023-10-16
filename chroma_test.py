import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv

# Initialize ChromaDB & Create Collection
chroma_db = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_db.get_or_create_collection(name="rfp_generator")

# Load Data
print(collection.peek())