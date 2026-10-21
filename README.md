# 🧠 Agentic RAG Knowledge Assistant

A 100% locally hosted, zero-API-cost Multi-Agent RAG (Retrieval-Augmented Generation) system. This project allows users to query complex PDF textbooks using natural language.

## 🚀 Features
- **Local LLMs:** Uses Ollama (`qwen2.5:1.5b`) and `nomic-embed-text` for completely offline embeddings and generation.
- **Multi-Agent Orchestration:** Uses AutoGen 0.4+ to create an `AnswerAgent` and a `CriticAgent` that converse to verify facts and eliminate AI hallucinations.
- **Custom Embedding Pipeline:** Bypasses LangChain wrapper bugs by making direct API calls to Ollama's embedding endpoint.
- **Interactive UI:** Deployed using Streamlit for a smooth user experience.

## 🛠️ Tech Stack
- **Framework:** AutoGen 0.4+
- **LLM & Embeddings:** Ollama
- **Vector Database:** ChromaDB
- **UI:** Streamlit
- **Language:** Python

## 💻 How to Run Locally
1. Install [Ollama](https://ollama.com/) and pull the required models:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
2. Clone this repository and install the Python requirements:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run streamlit_app.py
