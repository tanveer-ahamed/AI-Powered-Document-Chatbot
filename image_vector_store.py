#image_vector_store.py
import os, pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

clip_model = SentenceTransformer("clip-ViT-B-32")

def create_image_embeddings(images_meta, file_path="image_store.pkl"):
    img_paths = [m["path"] for m in images_meta]
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    embeddings = clip_model.encode(imgs, convert_to_numpy=True, show_progress_bar=True)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)

    data = {"images": images_meta, "embeddings": embeddings}
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved image store to {file_path}")

def load_image_store(file_path="image_store.pkl"):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def retrieve_images(query, store, top_k=5):
    q_vec = clip_model.encode([query], convert_to_numpy=True)[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    scores = store["embeddings"].dot(q_vec)
    top_idx = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_idx:
        img = store["images"][idx]
        results.append({
            "id": img["id"],
            "page": img["page"],
            "path": img["path"],
            "score": float(scores[idx])
        })
    return results
