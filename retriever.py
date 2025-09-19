#retriever.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
from vector_store import load_vector_store

# Models
TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CLIP_MODEL = SentenceTransformer("clip-ViT-B-32")

# Globals
_text_chunks, _text_embeddings = None, None
_image_store = None

# ----------- TEXT -----------
def init_text_store(vector_store_path="vector_store.pkl"):
    global _text_chunks, _text_embeddings
    _text_chunks, _text_embeddings = load_vector_store(vector_store_path)
    print(f"✅ Loaded text store ({len(_text_chunks)} chunks).")

def retrieve_text(query, top_k=5):
    if _text_chunks is None or _text_embeddings is None:
        raise ValueError("Text store not initialized.")
    q_vec = TEXT_MODEL.encode([query], convert_to_numpy=True)[0]
    q_vec /= (np.linalg.norm(q_vec) + 1e-10)
    scores = _text_embeddings.dot(q_vec)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [{
        "id": _text_chunks[i]["id"],
        "text": _text_chunks[i]["text"],
        "page": _text_chunks[i]["page"],
        "score": float(scores[i])
    } for i in top_idx]

# ----------- IMAGES -----------
def create_image_embeddings(images_meta, file_path="image_store.pkl"):
    img_paths = [m["path"] for m in images_meta]
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    embeddings = CLIP_MODEL.encode(imgs, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)
    data = {"images": images_meta, "embeddings": embeddings}
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved image store to {file_path}")

def load_image_store(file_path="image_store.pkl"):
    global _image_store
    with open(file_path, "rb") as f:
        _image_store = pickle.load(f)
    print(f"✅ Loaded image store ({len(_image_store['images'])} images).")

def retrieve_images(query, top_k=3):
    if _image_store is None:
        raise ValueError("Image store not initialized.")
    q_vec = CLIP_MODEL.encode([query], convert_to_numpy=True)[0]
    q_vec /= (np.linalg.norm(q_vec) + 1e-10)
    scores = _image_store["embeddings"].dot(q_vec)
    top_idx = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_idx:
        img = _image_store["images"][i]
        results.append({
            "id": img["id"],
            "page": img["page"],
            "path": img["path"],
            "filename": os.path.basename(img["path"]),  # ✅ Add this
            "score": float(scores[i])
        })
    return results
