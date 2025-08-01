# app.py
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from utils.pdf_processor import process_pdf_complete
from utils.vector_database import (
    initialize_embeddings,
    load_vector_database,
    add_to_vector_database,
    save_vector_database,
    get_database_stats
)
from utils.retrieval_qa import generate_medical_response, validate_medical_query

# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

client = Groq(api_key=GROQ_API_KEY)

# ─── File paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Medical_book.pdf")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, "medical_vector_store.faiss")
VECTOR_META_PATH = os.path.join(VECTOR_DB_DIR, "medical_vector_store.pkl")

# Ensure data directory exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ─── Initialize or Load Vector DB ─────────────────────────────────────────────
if os.path.exists(VECTOR_DB_PATH) and os.path.exists(VECTOR_META_PATH):
    # Load existing store
    embeddings, vector_db = load_vector_database(VECTOR_DB_PATH, VECTOR_META_PATH)
    print(f"[{datetime.now()}] Loaded vector DB with {get_database_stats(vector_db)['num_chunks']} chunks")
else:
    # Build store from PDF
    print(f"[{datetime.now()}] Vector DB not found, processing PDF...")
    chunks = process_pdf_complete(PDF_PATH)
    if not chunks:
        raise RuntimeError(f"Failed to process PDF at {PDF_PATH}")
    embeddings = initialize_embeddings()
    vector_db = add_to_vector_database(embeddings, chunks)
    save_vector_database(vector_db, VECTOR_DB_PATH, VECTOR_META_PATH)
    print(f"[{datetime.now()}] Created new vector DB with {get_database_stats(vector_db)['num_chunks']} chunks")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    stats = get_database_stats(vector_db)
    return jsonify({
        "status": "healthy",
        "num_chunks": stats["num_chunks"]
    })

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Validate content
    if not validate_medical_query(question):
        return jsonify({"error": "Inappropriate query"}), 400

    # Generate answer
    try:
        answer = generate_medical_response(client, embeddings, vector_db, question)
        return jsonify({
            "question": question,
            "answer": answer
        })
    except Exception as e:
        print(f"[{datetime.now()}] Error during response generation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # For local testing only; Vercel ignores this block
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
