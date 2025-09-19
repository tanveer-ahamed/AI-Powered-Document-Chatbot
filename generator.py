# generator.py
import os, json, datetime, pytz
import google.generativeai as genai
from retriever import retrieve_text, retrieve_images
from dotenv import load_dotenv

# ---------- Gemini setup ----------
load_dotenv(".env")
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in .env")

genai.configure(api_key=API_KEY)
_gemini_model = genai.GenerativeModel("gemini-1.5-flash")

LOG_FILE = "chat_logs.json"

# ---------- Helpers ----------
def safe_usage(obj):
    if obj is None:
        return None
    try:
        prompt_tokens = getattr(obj, "prompt_token_count", 0)
        candidates_tokens = getattr(obj, "candidates_token_count", 0)
        total_tokens = getattr(obj, "total_token_count", 0)
        return (
            f"prompt_token_count: {prompt_tokens}\n"
            f"candidates_token_count: {candidates_tokens}\n"
            f"total_token_count: {total_tokens}\n"
        )
    except Exception:
        try:
            return str(dict(obj))
        except Exception:
            return str(obj)

def log_interaction(entry: dict):
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# ---------- Query rewriting ----------
ACRONYM_DICT = {
    "system BMS": "System-level battery management system",
    "EMS": "Energy management system",
    "PCS": "Power conversion system",
    "xHMI": "External human machine interface",
    "UPS": "Uninterruptible power supply",
    "HVAC": "Heating, ventilation, air conditioning",
    "PBMS": "Pack-level battery management system",
    "string BMS": "String-level battery management system",
    "ESM": "Energy storage module",
    "FSS": "fire suppression system",
}

def rewrite_query(query: str) -> str:
    words = query.split()
    expanded = [ACRONYM_DICT.get(w.upper(), w) for w in words]
    return " ".join(expanded)

# ---------- Assemble context ----------
def assemble_context(text_chunks, image_chunks=None):
    text_part = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in text_chunks)
    if image_chunks:
        image_part = "\n".join(
            f"[Image {img['id']} on Page {img['page']}] (see {img['path']})"
            for img in image_chunks
        )
        return text_part + ("\n\n" + image_part if image_part else "")
    return text_part

# ---------- Gemini-based classification ----------
def classify_query_with_gemini(query: str) -> str:
    prompt = (
        "You are an AI assistant classifier. "
        "Classify the following user query into one of three categories:\n"
        "1. 'manual'  -> technical/manual question\n"
        "2. 'casual'  -> friendly greeting or small talk\n"
        "3. 'blocked' -> contains offensive or inappropriate content\n\n"
        f"User query: \"{query}\"\n\n"
        "Respond ONLY with one word: manual, casual, or blocked."
    )
    gen_cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=10)
    resp = _gemini_model.generate_content(prompt, generation_config=gen_cfg)
    classification = getattr(resp, "text", "").strip().lower()
    if classification not in {"manual", "casual", "blocked"}:
        classification = "manual"
    return classification

# ---------- Generate Response ----------
def generate_response(query: str, top_k_text: int = 10, top_k_img: int = 5):
    # ----- Exit check -----
    if query.strip().lower() == "exit":
        final_answer = raw_answer = "Goodbye!"
        timestamp = datetime.datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        log_entry = {
            "timestamp": timestamp,
            "user_query": query,
            "expanded_query": query,
            "retrieved_chunks": [],
            "images": [],
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "metrics": {"precision": 0.0, "recall": 0.0},
            "tokens_used": {"first_pass": None, "second_pass": None},
        }
        log_interaction(log_entry)
        return final_answer, log_entry

    # ----- Normal processing -----
    query_type = classify_query_with_gemini(query)
    query_expanded = rewrite_query(query)

    retrieved = []
    images = []
    precision = recall = 0.0
    resp1 = resp2 = None

    if query_type == "blocked":
        final_answer = raw_answer = (
            "The requested information is not available. Please consult the support team."
        )
    elif query_type == "casual":
        gen_cfg = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=50)
        resp = _gemini_model.generate_content(query, generation_config=gen_cfg)
        final_answer = raw_answer = getattr(resp, "text", "").strip() or "Hello!"
    else:  # manual
        retrieved = retrieve_text(query_expanded, top_k=top_k_text)
        images = retrieve_images(query_expanded, top_k=top_k_img)
        for img in images:
            img["filename"] = os.path.basename(img["path"])

        context = assemble_context(retrieved, images)
        relevant_retrieved = [r for r in retrieved if r.get("score", 0) >= 0.2]
        precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0.0
        recall = len(relevant_retrieved) / len(retrieved) if retrieved else 0.0

        # ---------- First pass ----------
        prompt1 = (
            "You are an assistant that must answer using ONLY the provided excerpts "
            "from the AmpD Enertainer User Manual (NCM). Do NOT use outside knowledge. "
            "If the manual doesn't contain the answer, respond exactly: "
            "'The manual does not contain this information.'\n\n"
            f"--- Manual Excerpts ---\n{context}\n\n"
            f"--- User Question ---\n{query}\n\n"
            "Instructions:\n"
            "- Summarize ALL relevant rules, cautions, and systems mentioned in the excerpts.\n"
            "- Do not omit any details, even if they appear repetitive.\n"
            "- Include page references for every point.\n"
            "- If multiple items are listed across pages, present them as a structured list.\n"
            "Answer:"
        )
        gen_cfg1 = genai.types.GenerationConfig(
            temperature=0.0, max_output_tokens=1000
        )
        resp1 = _gemini_model.generate_content(prompt1, generation_config=gen_cfg1)
        raw_answer = (
            getattr(resp1, "text", "").strip()
            or "The requested information is not available. Please consult the support team."
        )

        # ---------- Second pass: polish ----------
        prompt2 = (
            "Rewrite the following draft answer into a polished, clear, and complete response. "
            "Keep ALL technical details, page references, and image mentions intact. "
            "Do not shorten or omit any points.\n\n"
            f"--- Draft Answer ---\n{raw_answer}\n\n"
            "Final polished answer:"
        )
        gen_cfg2 = genai.types.GenerationConfig(
            temperature=0.0, max_output_tokens=500
        )
        resp2 = _gemini_model.generate_content(prompt2, generation_config=gen_cfg2)
        final_answer = getattr(resp2, "text", "").strip() or raw_answer

    # ----- Timestamp with HKT -----
    timestamp = datetime.datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    # ----- Log interaction -----
    log_entry = {
        "timestamp": timestamp,
        "user_query": query,
        "expanded_query": query_expanded,
        "retrieved_chunks": [
            {
                "id": r.get("id"),
                "text": r.get("text"),
                "page": r.get("page"),
                "score": r.get("score"),
            }
            for r in retrieved
        ],
        "images": images if query_type == "manual" else [],
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "metrics": {"precision": round(precision, 3), "recall": round(recall, 3)},
        "tokens_used": {
            "first_pass": safe_usage(getattr(resp1, "usage_metadata", None)),
            "second_pass": safe_usage(getattr(resp2, "usage_metadata", None)),
        },
    }
    log_interaction(log_entry)
    return final_answer, log_entry
