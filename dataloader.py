import chromadb
chroma_client = chromadb.Client()

sku_db = chroma_client.create_collection(name="sku")
