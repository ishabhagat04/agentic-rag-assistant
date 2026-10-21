# """
# Agentic RAG Knowledge Assistant — v2 (manual ChromaDB insertion)
# Bypasses LangChain's Chroma wrapper to guarantee embeddings work.
# """

# import asyncio
# import os
# import requests
# import chromadb

# from dotenv import load_dotenv

# # ---------- LangChain (only for PDF loading + chunking) ----------
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ---------- AutoGen 0.4+ ----------
# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.teams import RoundRobinGroupChat
# from autogen_agentchat.conditions import TextMentionTermination
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_core.models import ModelInfo

# load_dotenv()

# # ============================================================
# # CONFIG
# # ============================================================
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
# OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# PDF_PATH = "./docs/autogen_paper.pdf"
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "knowledge_base"
# QUESTION = "What is AutoGen and what are its main components?"


# # ============================================================
# # OLLAMA EMBEDDING — direct API call
# # ============================================================
# def ollama_embed(text: str) -> list:
#     """Call Ollama /api/embeddings and return the vector."""
#     resp = requests.post(
#         f"{OLLAMA_BASE_URL}/api/embeddings",
#         json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
#         timeout=120,
#     )
#     resp.raise_for_status()
#     data = resp.json()
#     if "embedding" not in data or not data["embedding"]:
#         raise RuntimeError(f"Empty embedding from Ollama for text: {text[:80]}...")
#     return data["embedding"]


# # ============================================================
# # STEP 1: BUILD VECTOR STORE (manual insertion)
# # ============================================================
# def build_retriever():
#     print("\n[1/4] Loading PDF and building vector store...")

#     # Load PDF
#     loader = PyPDFLoader(PDF_PATH)
#     documents = loader.load()
#     print(f"      Loaded {len(documents)} pages")

#     # Chunk
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100,
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"      Split into {len(chunks)} chunks")

#     # Setup ChromaDB client
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     try:
#         client.delete_collection(COLLECTION_NAME)
#     except Exception:
#         pass
#     collection = client.create_collection(
#         name=COLLECTION_NAME,
#         metadata={"hnsw:space": "cosine"},
#     )

#     # Manually embed each chunk and add to Chroma
#     print(f"      Embedding chunks (this takes a few minutes)...")
#     ids = []
#     texts = []
#     metadatas = []
#     embeddings = []

#     for i, chunk in enumerate(chunks):
#         vec = ollama_embed(chunk.page_content)
#         ids.append(f"chunk_{i}")
#         texts.append(chunk.page_content)
#         metadatas.append({"page": chunk.metadata.get("page", 0)})
#         embeddings.append(vec)
#         if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
#             print(f"      Embedded {i + 1}/{len(chunks)} chunks...")

#     collection.add(
#         ids=ids,
#         documents=texts,
#         metadatas=metadatas,
#         embeddings=embeddings,
#     )
#     print(f"      Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")

#     return collection


# # ============================================================
# # STEP 2: RETRIEVE CONTEXT
# # ============================================================
# def retrieve_context(collection, question: str, k: int = 3) -> str:
#     print(f"\n[2/4] Retrieving top {k} chunks for: {question}")

#     q_vec = ollama_embed(question)
#     results = collection.query(
#         query_embeddings=[q_vec],
#         n_results=k,
#     )

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]

#     context_parts = []
#     for doc, meta in zip(docs, metas):
#         page = meta.get("page", "?")
#         context_parts.append(f"[Source: page {page}]\n{doc}")

#     context = "\n\n---\n\n".join(context_parts)
#     print(f"      Retrieved {len(docs)} chunks")
#     return context


# # ============================================================
# # STEP 3: BUILD AGENTS
# # ============================================================
# def build_agents(context: str, question: str):
#     print("\n[3/4] Creating AutoGen agents...")

#     model_client = OpenAIChatCompletionClient(
#         model=OLLAMA_LLM_MODEL,
#         base_url=f"{OLLAMA_BASE_URL}/v1",
#         api_key="ollama",
#         model_info=ModelInfo(
#             vision=False,
#             function_calling=False,
#             json_output=False,
#             family="unknown",
#         ),
#     )

#     answer_agent = AssistantAgent(
#         name="AnswerAgent",
#         model_client=model_client,
#         system_message=(
#             "You are a research assistant. Answer the user's question using ONLY "
#             "the provided CONTEXT. Cite page numbers. "
#             "If the answer is not in the context, say: "
#             "'I cannot find the answer in the provided documents.'\n\n"
#             "End your reply with: DONE"
#         ),
#     )

#     critic_agent = AssistantAgent(
#         name="CriticAgent",
#         model_client=model_client,
#         system_message=(
#             "You are a fact-checker. Check that every claim in AnswerAgent's reply "
#             "is supported by the CONTEXT. If accurate, reply only: VERIFIED. "
#             "Otherwise, explain what's wrong and end with: DONE"
#         ),
#     )

#     team = RoundRobinGroupChat(
#         participants=[answer_agent, critic_agent],
#         termination_condition=TextMentionTermination("DONE"),
#         max_turns=4,
#     )

#     task = (
#         f"CONTEXT:\n{context}\n\n"
#         f"QUESTION: {question}\n\n"
#         f"AnswerAgent, please answer the question using the context above."
#     )

#     return team, task


# # ============================================================
# # STEP 4: RUN
# # ============================================================
# async def main():
#     collection = build_retriever()
#     context = retrieve_context(collection, QUESTION, k=3)
#     team, task = build_agents(context, QUESTION)

#     print("\n[4/4] Running agent conversation...\n" + "=" * 60)
#     result = await team.run(task=task)
#     print("=" * 60)

#     for msg in result.messages:
#         print(f"\n--- {msg.source} ---\n{msg.content}")

#     print("\n" + "=" * 60)
#     print("FINAL ANSWER")
#     print("=" * 60)
#     for msg in reversed(result.messages):
#         if msg.source == "AnswerAgent":
#             print(msg.content)
#             break


# if __name__ == "__main__":
#     asyncio.run(main())

"""
Agentic RAG Knowledge Assistant — v4 (Instant DB Load Fix)
Bypasses LangChain's Chroma wrapper to guarantee embeddings work.
Skips embedding if the ChromaDB collection already exists.
"""

import asyncio
import os
import requests
import chromadb

from dotenv import load_dotenv

# ---------- LangChain (only for PDF loading + chunking) ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- AutoGen 0.4+ ----------
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

PDF_PATH = "./docs/autogen_paper.pdf"  # Your Psychology book
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

QUESTION = "What is cognitive psychology and how does it compare to cognitive neuroscience?"


# ============================================================
# OLLAMA EMBEDDING — direct API call
# ============================================================
def ollama_embed(text: str) -> list:
    """Call Ollama /api/embeddings and return the vector."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if "embedding" not in data or not data["embedding"]:
        raise RuntimeError(f"Empty embedding from Ollama for text: {text[:80]}...")
    return data["embedding"]


# ============================================================
# STEP 1: BUILD VECTOR STORE (Instant Load Fix)
# ============================================================
def build_retriever():
    print("\n[1/4] Checking vector store...")

    # Setup ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        # If the collection already exists, just use it and skip embedding!
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            print(f"      Found existing ChromaDB with {collection.count()} chunks. Skipping embedding.")
            return collection
    except Exception:
        # If it doesn't exist, we build it from scratch below
        pass 

    print("      No existing database found. Building from PDF...")

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    all_documents = loader.load()
    
    # Limit to first 30 pages to save CPU time
    documents = all_documents[:30] 
    print(f"      Loaded {len(documents)} pages (limited to 30 for testing)")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"      Split into {len(chunks)} chunks")

    # Create a new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Manually embed each chunk and add to Chroma
    print(f"      Embedding chunks (this takes a few minutes)...")
    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        vec = ollama_embed(chunk.page_content)
        ids.append(f"chunk_{i}")
        texts.append(chunk.page_content)
        
        # PyPDFLoader page numbers are 0-indexed, so we add 1 for human-readable citations
        page_num = chunk.metadata.get("page", 0) + 1 
        metadatas.append({"page": page_num})
        
        embeddings.append(vec)
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"      Embedded {i + 1}/{len(chunks)} chunks...")

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"      Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")

    return collection


# ============================================================
# STEP 2: RETRIEVE CONTEXT
# ============================================================
def retrieve_context(collection, question: str, k: int = 3) -> str:
    print(f"\n[2/4] Retrieving top {k} chunks for: {question}")

    q_vec = ollama_embed(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(docs, metas):
        page = meta.get("page", "?")
        context_parts.append(f"[Source: page {page}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    print(f"      Retrieved {len(docs)} chunks")
    return context


# ============================================================
# STEP 3: BUILD AGENTS
# ============================================================
def build_agents(context: str, question: str):
    print("\n[3/4] Creating AutoGen agents...")

    model_client = OpenAIChatCompletionClient(
        model=OLLAMA_LLM_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family="unknown",
        ),
    )

    answer_agent = AssistantAgent(
        name="AnswerAgent",
        model_client=model_client,
        system_message=(
            "You are a research assistant. Answer the user's question using ONLY "
            "the provided CONTEXT. Cite page numbers. "
            "If the answer is not in the context, say: "
            "'I cannot find the answer in the provided documents.'\n\n"
            "End your reply with: DONE"
        ),
    )

    critic_agent = AssistantAgent(
        name="CriticAgent",
        model_client=model_client,
        system_message=(
            "You are a fact-checker. Check that every claim in AnswerAgent's reply "
            "is supported by the CONTEXT. If accurate, reply only: VERIFIED. "
            "Otherwise, explain what's wrong and end with: DONE"
        ),
    )

    team = RoundRobinGroupChat(
        participants=[answer_agent, critic_agent],
        termination_condition=TextMentionTermination("DONE"),
        max_turns=4,
    )

    task = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"AnswerAgent, please answer the question using the context above."
    )

    return team, task


# ============================================================
# STEP 4: RUN
# ============================================================
async def main():
    collection = build_retriever()
    context = retrieve_context(collection, QUESTION, k=3)
    team, task = build_agents(context, QUESTION)

    print("\n[4/4] Running agent conversation...\n" + "=" * 60)
    result = await team.run(task=task)
    print("=" * 60)

    for msg in result.messages:
        print(f"\n--- {msg.source} ---\n{msg.content}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    for msg in reversed(result.messages):
        if msg.source == "AnswerAgent":
            print(msg.content)
            break


if __name__ == "__main__":
    asyncio.run(main())