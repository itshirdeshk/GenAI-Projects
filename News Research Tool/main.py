import os
import streamlit as st
import time
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = "GEMINI_API_KEY"

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
file_path="vector_index.pkl"
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.7, top_p=0.85)

main_placeholder = st.empty()

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # create embeddings and save it to FAISS Index
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorindex_gemini = FAISS.from_documents(docs, gemini_embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_gemini, f)
        
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.subheader(result["answer"])
            
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)