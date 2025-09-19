#chatbot.py
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os, json

from pdf_processor import extract_text_by_page, extract_images
from vector_store import create_text_embeddings, save_vector_store
from retriever import init_text_store, create_image_embeddings, load_image_store
from generator import generate_response

app = Flask(__name__)
CORS(app)

PDF_PATH = "AmpD Enertainer User Manual (NCM) - Rev 2.3.pdf"
TEXT_STORE = "vector_store.pkl"
IMAGE_STORE = "image_store.pkl"
CHAT_LOG_FILE = "chat_logs.json"

# -------- Build vector stores if missing --------
if not os.path.exists(TEXT_STORE):
    print("⚡ Building text vector store...")
    pages = extract_text_by_page(PDF_PATH)
    chunks, embeddings = create_text_embeddings(pages)
    save_vector_store(chunks, embeddings, TEXT_STORE)

if not os.path.exists(IMAGE_STORE):
    print("⚡ Building image vector store...")
    images = extract_images(PDF_PATH)
    create_image_embeddings(images, IMAGE_STORE)

# -------- Initialize stores --------
init_text_store(TEXT_STORE)
load_image_store(IMAGE_STORE)

# -------- Chat history helper --------
def load_chat_history():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.route("/", methods=["GET", "POST"])
def home():
    chat_history = load_chat_history()
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            answer, log_entry = generate_response(user_query)
            chat_history.append(log_entry)
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, indent=2, ensure_ascii=False)
    return render_template("index.html", chat_history=chat_history)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.json or {}
    query = body.get("query", "")
    if not query:
        return jsonify({"error": "query required"}), 400
    answer, log_entry = generate_response(query)
    return jsonify({"answer": answer, "sources": log_entry})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
