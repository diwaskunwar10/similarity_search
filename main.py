import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import LongContextReorder
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_loaders import PyPDFLoader
import re
import streamlit as st

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Model Loaded..........")

st.title("PDF Analysis")
file_upload = st.file_uploader("Upload PDF", type=["pdf"])

if file_upload is not None:
    st.write("File uploaded successfully!")
    st.write("Ready for queries!")
    st.write("Type your query below:")
    uploaded_filename = "uploaded_file.pdf"  # Define a specific name for the uploaded file
    with open(uploaded_filename, "wb") as f:
        f.write(file_upload.getvalue())  # Save the uploaded file locally
    loader = PyPDFLoader(uploaded_filename)  # Pass the file path to PyPDFLoader
    match = re.search(r'([^/]+)\.pdf', uploaded_filename)
    if match:
        name = match.group(1)
    else:
        name = "Unknown"
    name = name
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    text_store = Chroma.from_documents(texts, hf, collection_metadata={"hsnw:space": "cosine"},
                                       persist_directory="data/facts")
    laod_store = Chroma(persist_directory="data/facts", embedding_function=hf)
    retreiver_store = laod_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    lotr = MergerRetriever(retrievers=[retreiver_store])
    query = st.text_input("Query:")

    ask_button = st.button("Ask")

    if ask_button:
            docs = lotr.get_relevant_documents(query)
            if len(docs) == 0:
                st.write("No relevant documents found for the query.")
            else:
                st.write("Relevant documents:",docs)

