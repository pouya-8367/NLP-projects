from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables for API keys
load_dotenv()

# Streamlit UI setup
st.title("Chat With Document")

# Initialize LLM with Google Gemini
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.2,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize embeddings with Ollama
embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"))

# File uploader for document input
upload_file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["txt", "pdf", "docx"])

# Main document processing pipeline
if upload_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=upload_file.name) as tmp_file:
        tmp_file.write(upload_file.read())
        file_path = tmp_file.name

    # Determine file type and select appropriate loader
    file_ext = Path(file_path).suffix

    if file_ext == ".txt":
        loader = TextLoader(file_path)
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file format")
        st.stop()

    try:
        # Show progress during document processing
        progress_bar = st.progress(0, text="Processing your document...")

        # Step 1: Load document
        progress_bar.progress(25, text="Loading file...")
        document = loader.load()

        # Step 2: Split document into semantic chunks
        progress_bar.progress(50, text="Splitting document into chunks...")
        chunker = SemanticChunker(embeddings=embeddings)
        chunks = chunker.split_documents(documents=document)

        # Step 3: Create vector embeddings and store in FAISS
        progress_bar.progress(75, text="Creating embeddings...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()

        progress_bar.progress(100, text="Document processing complete!")
        st.success("Document processed successfully!✅")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Clean up temporary file
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)

    # Define RAG prompt template for the assistant
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are Tech Haven's customer service assistant. "
         "Answer the user's question using only the provided context. "
         "Present information in a clear, structured way. "
         "Use line breaks to separate different points. "
         "Use bullet points for lists. "
         "If the context contains no relevant information, say: "
         "'Based on our policies, I don't have that specific information. "
         "Please contact support@techhaven.com for further assistance.'"
         ),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # Create RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

# Question input and answer generation
user_question = st.text_input("Ask a question about the document")

if user_question:
    # Generate answer using the RAG chain
    with st.spinner("Thinking...."):
        answer = chain.invoke(user_question).content
    
    # Display the answer
    st.write("### Answer:")
    st.write(answer)
