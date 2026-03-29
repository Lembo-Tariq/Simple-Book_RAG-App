# preprocess_alice.py
# Run this script ONCE to chunk and embed Alice in Wonderland and save
# the results to disk. After running this, the app loads Alice instantly
# from the saved files instead of reprocessing every time.
#
# Run it with:  python preprocess_alice.py

import json
import numpy as np
from ingestion import EMBEDDING_MODEL
from langchain.text_splitter import RecursiveCharacterTextSplitter

ALICE_PATH = "alice.txt"
CHUNKS_OUTPUT = "alice_chunks.json"
EMBEDDINGS_OUTPUT = "alice_embeddings.npy"

print("[Preprocess] Reading alice.txt...")
with open(ALICE_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()
# open() + f.read() loads the entire alice.txt file into one long string.

print("[Preprocess] Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", "!", "?", ".", " ", ""],
)
chunks = splitter.split_text(raw_text)
print(f"[Preprocess] {len(chunks)} chunks created.")

print("[Preprocess] Embedding chunks (this is the slow part, only happens once)...")
embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)
# encode(chunks) returns a NumPy array of shape (num_chunks, 384).
# This is the step that takes time — running all chunks through the
# neural network. After this script, we never need to do it again.

print("[Preprocess] Saving chunks to alice_chunks.json...")
with open(CHUNKS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
# json.dump(chunks, f) converts the Python list of strings into JSON
# format and writes it to alice_chunks.json.
# ensure_ascii=False preserves special characters like curly quotes.
# indent=2 makes the file human-readable with 2-space indentation.

print("[Preprocess] Saving embeddings to alice_embeddings.npy...")
np.save(EMBEDDINGS_OUTPUT, embeddings)
# np.save("alice_embeddings.npy", embeddings) writes the NumPy array
# to a binary .npy file. Loading it back with np.load() takes milliseconds
# regardless of how large the array is.

print("[Preprocess] Done! You can now run the app with: streamlit run app.py")
