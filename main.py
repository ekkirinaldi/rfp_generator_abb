import streamlit as st
import chromadb
from vectorizer import vectorize
from chromadb.utils import embedding_functions
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()
chroma_db = chromadb.PersistentClient(path="vectordb")

# Data Loaders
data = vectorize('Dataset-small.csv')

# Generate Embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name = "text-embedding-ada-002"
)

llm = OpenAI(temperature=0, model="text-embedding-ada-002")

# Access collection
collection = chroma_db.get_collection(name="rfp_generator", embedding_function=openai_ef)

# Setup LLMChain, Prompts, Template
systemtemplate="""
You are quotation generator. Your job is to reply the email politely and create a table of items requested including the price, quantity, price per unit, and total cost.

{context}

Question: {question}

When you write the email, please include the following information:
1. Make a table when listing the items and their prices.
2. Each item should have a quantity, price per unit, and total cost.
3. Each inquiry should include shipping cost. Shipping cost is 15 percent from total cost.
4. Include tax, which 10 percent from total cost.
5. Put tax and shipping cost in the last row of the table.
6. Information of sender: Annisa Mahendra, VP of Operations, ABB Inc., email: annisa@abb.com, phone: 1234567890
7. Each row should unique item. If there are multiple items with the same StockCode, please combine them into one row.
8. If the item is not available, tell the customer that the item is not available.
"""

# Generate Context
def retrieve_context(query):
    similar_responses = collection.query(
        query_texts=[query],
        n_results=3
    )
    context = similar_responses['documents']
    return context

chatllm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
prompt = PromptTemplate(template=systemtemplate, input_variables=["context", "question"])
chain = LLMChain(llm=chatllm, prompt=prompt)

# query = """
# Dear Supplier,

# I hope this email finds you well. I am interested in placing a bulk order for clocks and came across your esteemed establishment as a reputable supplier within the UK market.

# I am looking to purchase 10 units each of the 6 cheapest clock models available in your collection. It would be greatly appreciated if you could provide a quotation including the unit price, total cost, and any available discounts for bulk purchases.

# Moreover, I would also like to inquire about the estimated delivery time and shipping charges to Jakarta.

# I am keen to establish a fruitful business relationship and am hopeful for favorable pricing and timely delivery. I am available for further discussion at your earliest convenience. You can reach me via email or at 0987654321.

# Thank you for your prompt attention to this matter. I look forward to hearing from you soon.

# Warm regards,

# Reynold Smith
# VP of Operations
# Good Company Inc.
# """

# response = chain.run(context=context, question=query)

def main():
    st.set_page_config(
        page_title="ABB RFQ Generator", page_icon="🤖", layout="centered", initial_sidebar_state="auto"
    )
    st.header("ABB RFQ Generator")
    message = st.text_area("Example: 'I want to order the 6 cheapest clocks from the UK. I want to order 10 each'")

    if message:
        st.write("Answer:")
        context = retrieve_context(message)
        result = chain.run(context=context, question=message)
        st.info(result)
        
if __name__ == "__main__":
    main()