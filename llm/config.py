"""
Configuration constants for the LLM module.
Ollama settings, model parameters, and system prompts are centralized here.
"""

# ──────────────────────────────────────────────
# Ollama Server
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_CHAT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"

# ──────────────────────────────────────────────
# Model Settings
# ──────────────────────────────────────────────
DEFAULT_MODEL = "llama3"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Nexus AI, a helpful, accurate, and concise assistant. "
    "You provide clear, well-structured answers. "
    "When you don't know something, you say so honestly. "
    "Keep responses informative but not overly verbose."
)

# ──────────────────────────────────────────────
# Conversation Settings
# ──────────────────────────────────────────────
MAX_HISTORY_TURNS = 20      # Maximum conversation turns to send as context
REQUEST_TIMEOUT = 120       # Seconds to wait for Ollama response
