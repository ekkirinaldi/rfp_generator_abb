import streamlit as st
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# 1. Data Loaders
loader = CSVLoader('Dataset-small.csv')
data = loader.load()

# 2. Split Texts into Embeddings
indexer = VectorstoreIndexCreator()
docsearch = indexer.from_documents(data)

# 3. Setup LLMChain, Prompts, Template
systemtemplate="""
You are proposal generator for ecommerce. Your job is to reply the email and create a quotation from list of items requested including the price, quantity, price per unit, and total cost.

{context}

Question: {question}
writte the answer in markdown format
Make a table when listing the items and their prices.
Each item should have a quantity, price per unit, and total cost.
Each inquiry should include shipping cost. Shipping cost is 15 percent from total cost.
Include tax, which 10 percent from total cost.
put tax and shipping cost in the last row of the table.
"""
prompt = PromptTemplate(template=systemtemplate, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": prompt}
chain = RetrievalQA.from_chain_type(llm=OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()]), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)

# query = """
# Hi,
# I want to buy 10 alarm clocks, how much is it?
# """

# response = chain.run(query)

def main():
    st.set_page_config(
        page_title="RFP Generator", page_icon="🤖", layout="centered", initial_sidebar_state="auto"
    )
    st.header("RFP Generator")
    message = st.text_area("Example: 'Who is Elfie.co founders?', 'how big is their team?', 'what is elfie features?'")
    
    if message:
        st.write("Answer:")
        result = chain.run(message)
        st.info(result)
        
if __name__ == "__main__":
    main()