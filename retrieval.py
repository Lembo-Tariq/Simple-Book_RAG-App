from ingestion import EMBEDDING_MODEL
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


def retrieve_and_answer(query: str, collection: chromadb.Collection) -> dict:
    query_embedding = EMBEDDING_MODEL.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant answering questions about a book.
Use ONLY the context provided below to answer. Be detailed and descriptive -
explain what the context reveals, do not just name things. Use full sentences.
If the answer is not in the context, say "I could not find that information in the document."

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }
