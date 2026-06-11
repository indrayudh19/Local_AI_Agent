"""
Flask backend API for the Nexus AI chatbot.
Serves the static frontend and exposes REST endpoints for the RAG pipeline.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from rag.pipeline import RAGPipeline
from rag.config import UPLOADS_DIR
from llm.orchestrator import LLMOrchestrator
from deepresearch_agent.deep_research import DeepResearchAgent

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {"pdf"}

# Initialize the RAG pipeline (loads models on first query)
print("[App] Initializing RAG pipeline...")
pipeline = RAGPipeline()
print("[App] Initializing LLM orchestrator...")
llm_orchestrator = LLMOrchestrator()
print("[App] Initializing Deep Research agent...")
deep_research_agent = DeepResearchAgent()
print("[App] Server ready.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
# Static Frontend
# ──────────────────────────────────────────────

@app.route("/")
def serve_index():
    """Serve the main frontend page."""
    return send_from_directory(app.static_folder, "index.html")


# ──────────────────────────────────────────────
# Chat API
# ──────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle a chat message.

    Request JSON:
        { "message": "What is the main topic of the document?" }

    Response JSON:
        {
            "answer": "...",
            "sources": [{"file": "...", "page": 1}],
            "confidence": 0.65,
            "confidence_label": "high"
        }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = pipeline.query(message)
        return jsonify(result)
    except Exception as e:
        print(f"[App] Error during query: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# ──────────────────────────────────────────────
# Document Upload API
# ──────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Upload and ingest a PDF document.

    Request: multipart/form-data with a 'file' field.

    Response JSON:
        { "filename": "...", "chunks": 42, "status": "success" }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)

    # Save the uploaded file
    file.save(filepath)
    print(f"[App] Saved uploaded file: {filepath}")

    # Ingest into the RAG pipeline
    try:
        result = pipeline.ingest(filepath)
        return jsonify(result)
    except Exception as e:
        print(f"[App] Error during ingestion: {e}")
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


# ──────────────────────────────────────────────
# Document Management API
# ──────────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
def list_documents():
    """
    List all ingested documents.

    Response JSON:
        { "documents": ["file1.pdf", "file2.pdf"] }
    """
    docs = pipeline.list_documents()
    return jsonify({"documents": docs})


@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    """
    Delete a document from the system.

    Response JSON:
        { "filename": "...", "status": "deleted" }
    """
    try:
        result = pipeline.delete_document(filename)
        return jsonify(result)
    except Exception as e:
        print(f"[App] Error during deletion: {e}")
        return jsonify({"error": f"Deletion failed: {str(e)}"}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Get system statistics.

    Response JSON:
        { "total_vectors": 100, "total_documents": 2, "documents": [...] }
    """
    return jsonify(pipeline.get_stats())


# ──────────────────────────────────────────────
# LLM API (Ollama-backed)
# ──────────────────────────────────────────────

@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    """
    Handle a chat message via the local LLM (Ollama).

    Request JSON:
        {
            "message": "Hello, how are you?",
            "history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        }

    Response JSON:
        { "answer": "...", "sources": [], "confidence": 1.0, "mode": "llm" }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    history = data.get("history", [])

    try:
        result = llm_orchestrator.chat(message, history=history)
        return jsonify(result)
    except Exception as e:
        print(f"[App] LLM error: {e}")
        return jsonify({"error": f"LLM error: {str(e)}"}), 500


@app.route("/api/llm/health", methods=["GET"])
def llm_health():
    """
    Check if the Ollama backend is available and the model is ready.

    Response JSON:
        { "status": "ok", "ollama_running": true, "model_available": true, ... }
    """
    return jsonify(llm_orchestrator.check_health())


@app.route("/api/llm/models", methods=["GET"])
def llm_models():
    """
    List available Ollama models.

    Response JSON:
        { "models": [...], "status": "ok" }
    """
    return jsonify(llm_orchestrator.list_models())


# ──────────────────────────────────────────────
# Deep Research API
# ──────────────────────────────────────────────

@app.route("/api/deep_research", methods=["POST"])
def deep_research():
    """
    Handle a deep research query.

    Request JSON:
        { "message": "Research the latest advancements in solid state batteries" }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = deep_research_agent.research(message)
        # To distinguish modes on the frontend
        result["mode"] = "deep_research"
        return jsonify(result)
    except Exception as e:
        print(f"[App] Deep Research error: {e}")
        return jsonify({"error": f"Deep Research error: {str(e)}"}), 500


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
