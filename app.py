import streamlit as st
import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_community.llms import HuggingFaceHub
import tempfile
from urllib.parse import urlparse
import time
st.set_page_config(
    page_title="Context-Aware Chatbot Developer Muhammad Waqar",
    page_icon="🤖",
    layout="wide"
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def load_documents(source):
    documents = []
    if isinstance(source, list) and all(hasattr(item, 'name') for item in source):  # Uploaded files
        for uploaded_file in source:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
                documents.extend(loader.load())
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")

            os.unlink(tmp_file_path)
    
    elif isinstance(source, str) and is_valid_url(source):  # URL
        try:
            loader = WebBaseLoader(source)
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading URL: {e}")
            return None
    
    elif isinstance(source, str):  # Direct text input
        from langchain.schema import Document
        documents = [Document(page_content=source)]
    
    return documents

def process_documents(documents):
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    repo_id = "google/flan-t5-base"
    
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.7, "max_length": 512},
        huggingfacehub_api_token=st.secrets.get("HF_API_KEY", None)
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True
    )
    
    return qa_chain
def simple_qa(question, documents, chat_history):
   
    doc_text = " ".join([doc.page_content for doc in documents])
    response = f"I found information related to your question. Based on the documents, here's what I know: {doc_text[:500]}..."
    
    if "summary" in question.lower():
        response = f"Here's a summary of the documents: {doc_text[:800]}..."
    elif "what" in question.lower() or "how" in question.lower():
        response = f"According to the documents: {doc_text[:600]}..."
    
    return response, documents[:2] 

st.title("🤖 Context-Aware Chatbot")
st.markdown("""
This chatbot uses open-source models and Retrieval-Augmented Generation (RAG) to answer questions 
based on provided documents while maintaining conversation context. No API keys required!
""")

with st.sidebar:
    st.header("Knowledge Source")
    source_option = st.radio(
        "Select knowledge source:",
        ["Upload Files", "Enter URL", "Enter Text"]
    )
    
    source = None
    if source_option == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        source = uploaded_files
        
    elif source_option == "Enter URL":
        url = st.text_input("Enter URL")
        if url:
            source = url
            
    elif source_option == "Enter Text":
        text_input = st.text_area("Enter text content")
        if text_input:
            source = text_input
    
    # Process button
    process_btn = st.button("Process Documents")
    
    if process_btn:
        if not source:
            st.error("Please provide a knowledge source")
        else:
            with st.spinner("Processing documents..."):
                documents = load_documents(source)
                if documents:
                    try:
                
                        st.session_state.qa_chain = process_documents(documents)
                        st.session_state.processed = True
                        st.session_state.documents = documents
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.warning(f"Using simplified mode: {str(e)}")
                    
                        st.session_state.processed = True
                        st.session_state.documents = documents
                        st.session_state.simple_mode = True
                        st.success("Documents processed successfully (using simplified mode)!")
                else:
                    st.error("Failed to load documents")

if st.session_state.processed and "documents" in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                if hasattr(st.session_state, 'simple_mode') and st.session_state.simple_mode:
                    response, source_documents = simple_qa(
                        prompt, 
                        st.session_state.documents, 
                        st.session_state.chat_history
                    )
                    message_placeholder.markdown(response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                   
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(source_documents):
                            st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                else:
                    result = st.session_state.qa_chain({"question": prompt})
                    response = result["answer"]
                    message_placeholder.markdown(response)
                    
                   
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                   
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try a different question or re-process your documents.")
    
    if st.button("Clear Conversation"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("👈 Please provide a knowledge source in the sidebar to start chatting.")
    
    
    st.markdown("""
    ### How to use:
    1. Choose a knowledge source:
        - **Upload Files**: PDF or text documents
        - **Enter URL**: A webpage URL to scrape content from
        - **Enter Text**: Directly paste text content
    2. Click "Process Documents" to create the knowledge base
    3. Start asking questions in the chat interface!
    
    ### Example documents you can try:
    - Copy and paste text from Wikipedia articles
    - Upload research papers or articles in PDF format
    - Provide URLs of informative web pages
    
    ### Example questions:
    - "What are the main points discussed in the document?"
    - "Can you summarize the key findings?"
    - "Explain the main concepts based on the provided information"
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "Free Context-Aware Chatbot using Open-Source Models | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)