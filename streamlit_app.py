import streamlit as st
import asyncio
import os
import requests
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

PDF_PATH = "./docs/autogen_paper.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

# ============================================================
# CORE FUNCTIONS
# ============================================================
def ollama_embed(text: str) -> list:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def get_collection(force_rebuild=False):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0 and not force_rebuild:
            return collection, "Loaded existing DB"
    except Exception:
        pass
        
    # Build from scratch
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()[:30] # Limit to 30 pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    ids, texts, metadatas, embeddings = [], [], [], []
    for i, chunk in enumerate(chunks):
        vec = ollama_embed(chunk.page_content)
        ids.append(f"chunk_{i}")
        texts.append(chunk.page_content)
        metadatas.append({"page": chunk.metadata.get("page", 0) + 1})
        embeddings.append(vec)
    
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return collection, "Built new DB"

def retrieve_context(collection, question: str, k: int = 3) -> str:
    q_vec = ollama_embed(question)
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_parts = [f"[Source: page {meta.get('page', '?')}]\n{doc}" for doc, meta in zip(docs, metas)]
    return "\n\n---\n\n".join(context_parts)

def build_agents(context: str, question: str):
    # Import AutoGen inside the function to prevent initial load crashes
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo

    model_client = OpenAIChatCompletionClient(
        model=OLLAMA_LLM_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        model_info=ModelInfo(vision=False, function_calling=False, json_output=False, family="unknown"),
    )

    answer_agent = AssistantAgent(
        name="AnswerAgent", model_client=model_client,
        system_message="Answer ONLY using the CONTEXT. Cite pages. End with DONE.",
    )

    critic_agent = AssistantAgent(
        name="CriticAgent", model_client=model_client,
        system_message="Check AnswerAgent's facts against CONTEXT. If accurate, reply VERIFIED. Otherwise explain and end with DONE.",
    )

    team = RoundRobinGroupChat(
        participants=[answer_agent, critic_agent],
        termination_condition=TextMentionTermination("DONE"), max_turns=4,
    )

    task = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswerAgent, please answer."
    return team, task

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠")
st.title("🧠 Agentic RAG Knowledge Assistant")
st.write("Ask a question based on the Psychology Handbook.")

with st.sidebar:
    st.header("Settings")
    if st.button("Force Rebuild Database"):
        with st.spinner("Rebuilding (takes a few minutes)..."):
            get_collection(force_rebuild=True)
            st.success("Database rebuilt!")

question = st.text_input("Enter your question:", "What is cognitive psychology?")

if st.button("Ask Agents"):
    if question:
        with st.spinner("Connecting to database..."):
            collection, status = get_collection()
            st.caption(f"Status: {status} ({collection.count()} chunks)")
        
        with st.spinner("Agents are thinking and verifying..."):
            context = retrieve_context(collection, question, k=3)
            team, task = build_agents(context, question)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(team.run(task=task))
            
            st.subheader("Final Answer:")
            for msg in result.messages:
                if msg.source == "AnswerAgent":
                    st.success(msg.content)
                    break
            
            with st.expander("See Agent Conversation Log"):
                for msg in result.messages:
                    st.write(f"**{msg.source}:** {msg.content}")