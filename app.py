import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit UI
st.title("Enterprise Document Q&A Bot 🤖")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split the PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Convert to embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Setup RetrievalQA chain
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever())

    # Chat with document
    question = st.text_input("Ask a question about the document:")
    if question:
        response = qa.run(question)
        st.write("Answer:", response)
