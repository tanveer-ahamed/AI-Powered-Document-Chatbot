#vector_store.py
import re
import textwrap
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Text embedding model ----------
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
_text_model = SentenceTransformer(TEXT_MODEL_NAME)

def split_into_chunks_by_page(pages, max_chars=1500, overlap=200):
    """
    Splits page texts into manageable chunks with optional overlap.
    Returns list of dicts: {"id", "text", "page"}
    """
    chunks = []
    chunk_id = 0
    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for para in paragraphs:
            if len(para) <= max_chars:
                chunks.append({"id": chunk_id, "text": para, "page": page_num})
                chunk_id += 1
            else:
                sub_chunks = textwrap.wrap(para, max_chars,
                                           break_long_words=False,
                                           replace_whitespace=False)
                for i, sub in enumerate(sub_chunks):
                    if i > 0 and overlap:
                        prev = sub_chunks[i-1]
                        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                        merged = overlap_text + " " + sub
                        chunks.append({"id": chunk_id, "text": merged, "page": page_num})
                    else:
                        chunks.append({"id": chunk_id, "text": sub, "page": page_num})
                    chunk_id += 1
    return chunks

def create_text_embeddings(pages, max_chars=1500, overlap=200):
    """
    Returns chunks + embeddings for text pages.
    - chunks: list of dicts {"id", "text", "page"}
    - embeddings: numpy array (n_chunks, dim)
    """
    chunks = split_into_chunks_by_page(pages, max_chars=max_chars, overlap=overlap)
    texts = [c["text"] for c in chunks]
    embeddings = _text_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)
    return chunks, embeddings

def save_vector_store(chunks, embeddings, file_path="vector_store.pkl"):
    """
    Save chunks and embeddings to pickle.
    """
    with open(file_path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"✅ Saved text vector store to {file_path}")

def load_vector_store(file_path="vector_store.pkl"):
    """
    Load chunks + embeddings from pickle.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]
