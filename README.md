## 📌 Objective
Building a prototype for an **AI-powered chat experience** 

- The primary goal of the chat experience is to answer customer questions about their equipment by leveraging the **PDF User Manual**.  
- The chatbot should **intelligently leverage** both text and images from the manual, retrieving relevant context and generating coherent responses.  
- Retrieval-Augmented Generation (RAG) is used to ensure responses are grounded in the provided document.  

---
## Programming language:
- Python **3.9+**
---

## ✅ Requirements & Implementation

# Place the PDF document in the ROOT folder 
# Mention the PDF document PATH in pdf_processor.py and chatbot.py

### 1. Document Processing
- Extracts **text** and **images** from the provided PDF using `PyMuPDF`.  
- Text embeddings are created with **SentenceTransformers (all-MiniLM-L6-v2)**.  
- Image embeddings are created with **clip-ViT-B-32**.  
- Results are stored in:
vector_store.pkl # text embeddings
image_store.pkl # image embeddings

- Extracted images are saved in:
static/images/

---

### 2. Query Handling & Retrieval
- Users ask questions via a simple **Flask web interface** (`index.html`).  
- The retriever searches for **relevant text chunks** and **images**.  
- Basic error handling ensures:
  
---

### 3. Response Generation
- Responses are generated with **Google Gemini API** (**gemini-1.5-flash** model)
- The system classifies queries into:
- **manual** → answers grounded in the PDF.  
- **casual** → friendly chatbot-like replies.  
- **blocked** → politely refuse inappropriate queries.  

---

### 4. Chat Interface
- A **minimal web UI** built with Flask + HTML template.  
- Allows users to input questions and view responses.  
- Chat history is stored in:
  chat_logs.json

- Screenshots / images related to the chatbot can be saved in:
results

---

### 5. Constraints & Best Practices
- **Efficiency**:
- Embeddings are **precomputed** on the first run and saved for reuse.  
 
- **Code Quality**:
- Modularized: separate modules for processing, retrieval, and generation.  
- Error handling and logging included.  
- **Documentation**:
- This `README.md` outlines the design, requirements, and setup steps.  

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create venv (Python 3.9+ recommended)
python -m venv venv

# Activate venv
# On Linux / macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add API Key
Create a .env file with:

```ini
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Running the Chatbot
Start the chatbot:

```bash
python chatbot.py
```

### First Run:
- Extracts text + images from PDF.  
- Creates embeddings → `vector_store.pkl`, `image_store.pkl`.  
- Saves images in `static/images/`.  
- ⚠️ **This step takes a little time.**

### Subsequent Runs:
- Loads precomputed embeddings → much faster startup.  
- Opens at 👉 http://127.0.0.1:5000/  

---

## 📂 Project Structure
```
.
├── chatbot.py             # Flask app entry point
├── generator.py           # Response generation w/ Gemini
├── retriever.py           # Text + image retrievers
├── pdf_processor.py       # PDF text & image extraction
├── vector_store.py        # Text embeddings logic
├── image_vector_store.py  # Image embeddings logic
├── static/images/         # Extracted images
├── templates/index.html   # Simple chat UI
├── results/               # Output folder
│   ├── chatbot_images/    # chatbot screenshots/images
│   
├── vector_store.pkl       # Generated text embeddings
├── image_store.pkl        # Generated image embeddings
├── requirements.txt       # Dependencies
├── chat_logs.json         # chat history
└── .env                   # API key
```

---

## 📝 Example Flow
User runs:

```bash
python chatbot.py
```

System builds embeddings (**first run only**).  

User opens browser at `127.0.0.1:5000`.  


Chatbot retrieves relevant excerpts + diagrams from the manual.  
Chatbot generates a clear, grounded answer with Gemini.  
Logs are stored in `chat_logs.json`.  

---

## 🎯 Deliverables
- Working chatbot prototype (**Flask app**)  
- Source code + `requirements.txt`  
- This `README.md` (system design + setup instructions)  
- `results/` folder containing:
  - `chat_logs.json`  
  - `chatbot_images/` (supporting images)  
