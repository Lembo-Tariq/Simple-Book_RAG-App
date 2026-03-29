from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import json
import numpy as np

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def _fresh_collection(client: chromadb.EphemeralClient) -> chromadb.Collection:
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass
    return client.create_collection(name="rag_collection")


def ingest_text(raw_text: str) -> chromadb.Collection:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "!", "?", ".", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True).tolist()

    client = chromadb.EphemeralClient()
    collection = _fresh_collection(client)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )
    return collection


def load_precomputed_alice(chunks_path: str, embeddings_path: str) -> chromadb.Collection:
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(embeddings_path).tolist()

    client = chromadb.EphemeralClient()
    collection = _fresh_collection(client)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )
    return collection
