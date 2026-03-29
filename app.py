import streamlit as st
import os
from ingestion import ingest_text, load_precomputed_alice
from retrieval import retrieve_and_answer

st.set_page_config(page_title="Book RAG App", page_icon="📚", layout="centered")
st.title("📚 Book Q&A — RAG App")
st.caption("Upload a text file or use Alice in Wonderland, then ask questions about it.")

if "collection" not in st.session_state:
    st.session_state.collection = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if not st.session_state.document_loaded:
    st.subheader("Step 1 — Choose a document")

    source_choice = st.radio(
        label="Where is your document?",
        options=["Upload my own .txt file", "Use Alice in Wonderland (built-in)"],
    )

    if source_choice == "Upload my own .txt file":
        st.info(
            "💡 **Looking for a book to try?** "
            "[Project Gutenberg](https://www.gutenberg.org) offers thousands of free classic books "
            "in plain text format — no sign-up required. Just search for your favourite title, "
            "download the **Plain Text UTF-8** version, and upload it here. "
            "Great options: Sherlock Holmes, Moby Dick, Pride and Prejudice, and more."
        )
        uploaded_file = st.file_uploader(label="Upload a plain text file (.txt)", type=["txt"])

        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8")
            if st.button("Load & Process Document"):
                with st.spinner("Chunking and embedding your document..."):
                    st.session_state.collection = ingest_text(raw_text)
                    st.session_state.document_loaded = True
                    st.session_state.document_name = uploaded_file.name
                st.rerun()
    else:
        CHUNKS_PATH = "alice_chunks.json"
        EMBEDDINGS_PATH = "alice_embeddings.npy"
        precomputed_ready = os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH)

        if precomputed_ready:
            st.info("✅ Alice in Wonderland is pre-processed and ready — loads instantly!")
            if st.button("Load Alice in Wonderland"):
                with st.spinner("Loading pre-computed Alice in Wonderland..."):
                    st.session_state.collection = load_precomputed_alice(CHUNKS_PATH, EMBEDDINGS_PATH)
                    st.session_state.document_loaded = True
                    st.session_state.document_name = "Alice in Wonderland"
                st.rerun()
        elif os.path.exists("alice.txt"):
            st.warning("alice.txt found but not pre-processed. Run: python preprocess_alice.py")
        else:
            st.warning(
                "alice.txt not found. Download from https://www.gutenberg.org/ebooks/11 "
                "(Plain Text UTF-8), save as alice.txt, then run: python preprocess_alice.py"
            )
else:
    st.success(f"✅ Loaded: **{st.session_state.document_name}**")

    if st.button("🔄 Load a different document"):
        st.session_state.collection = None
        st.session_state.document_loaded = False
        st.session_state.document_name = ""
        st.rerun()

    st.subheader("Step 2 — Ask a question")
    query = st.text_input(
        label="Type your question about the document:",
        placeholder="e.g. Who is the Queen of Hearts?"
    )

    if st.button("Get Answer") and query.strip():
        with st.spinner("Searching document and generating answer..."):
            result = retrieve_and_answer(query, st.session_state.collection)

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("📄 View source chunks used to generate this answer"):
            for i, chunk in enumerate(result["sources"]):
                st.markdown(f"**Chunk {i + 1}:**")
                st.text(chunk)
                st.divider()
