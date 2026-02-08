# RAG Document Chat

Ask questions about your documents (PDF, DOCX, TXT) using Retrieval-Augmented Generation with Google Gemini and Ollama embeddings.

## ✨ Features
- Upload PDF, DOCX, or TXT files
- Semantic chunking for better context retention
- FAISS vector store for fast retrieval
- Streamlit UI for easy interaction
- Secure API key management via `.env`

## 🚀 Setup


### Installation
```bash
# Clone repo
git clone https://github.com/pouya-abdoli/NLP-projects.git
cd NLP-projects/rag-document-chat

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env
```

### Configure `.env`
Edit `.env` with your credentials:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_MODEL=nomic-embed-text
```

### Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
rag-document-chat/
├── app.py          # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env.example    # Environment variables template
└── .gitignore      # Excludes secrets/cache files
└── README.md            # This file

```


## ⚙️ Tech Stack
- **LLM**: Google Gemini (`langchain-google-genai`)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Vector DB**: FAISS
- **UI**: Streamlit
- **Chunking**: SemanticChunker (`langchain-experimental`)

