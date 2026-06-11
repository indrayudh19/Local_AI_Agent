"""
LLM Orchestrator — the main entry point for LLM-based conversations.

Uses Ollama's REST API to communicate with locally-running language models.
Manages agent routing, health checks, and error handling.
"""

import requests

from llm.config import (
    OLLAMA_API_CHAT,
    OLLAMA_API_TAGS,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    REQUEST_TIMEOUT,
)
from llm.agents import ConversationAgent


class LLMOrchestrator:
    """
    Main orchestrator for LLM-based chat.

    Manages:
    - Communication with the Ollama API
    - Agent selection and routing
    - Health checking and model listing
    - Graceful error handling

    Usage:
        orchestrator = LLMOrchestrator()
        result = orchestrator.chat("Hello, how are you?", history=[])
        print(result["answer"])
    """

    def __init__(self, model=None):
        self.model = model or DEFAULT_MODEL
        self.conversation_agent = ConversationAgent()
        print(f"[LLMOrchestrator] Initialized with model: {self.model}")

    def chat(self, message, history=None):
        """
        Send a message to the local LLM and get a response.

        Args:
            message: The user's message string.
            history: Optional list of previous messages:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            dict with "answer", "sources", "confidence", "confidence_label", "mode"
        """
        # Build the full message context
        messages = self.conversation_agent.build_messages(message, history)

        try:
            # Call the Ollama chat API (non-streaming)
            response = requests.post(
                OLLAMA_API_CHAT,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                    }
                },
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 404:
                return {
                    "answer": (
                        f"Model '{self.model}' not found. "
                        f"Please run `ollama pull {self.model}` to download it, "
                        f"or set a different model in llm/config.py."
                    ),
                    "sources": [],
                    "confidence": 0.0,
                    "confidence_label": "none",
                    "mode": "llm"
                }

            response.raise_for_status()
            data = response.json()

            # Extract the assistant's reply
            raw_answer = data.get("message", {}).get("content", "")
            if not raw_answer:
                raw_answer = "I received an empty response. Please try again."

            return self.conversation_agent.format_response(raw_answer)

        except requests.exceptions.ConnectionError:
            return {
                "answer": (
                    "⚠️ Cannot connect to Ollama. Please make sure Ollama is running.\n\n"
                    "To start Ollama:\n"
                    "1. Install from https://ollama.com\n"
                    "2. Run `ollama serve` in a terminal\n"
                    f"3. Pull a model: `ollama pull {self.model}`"
                ),
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none",
                "mode": "llm"
            }

        except requests.exceptions.Timeout:
            return {
                "answer": (
                    "⚠️ The LLM took too long to respond. "
                    "This could happen with large models on limited hardware. "
                    "Try using a smaller model or simplifying your question."
                ),
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none",
                "mode": "llm"
            }

        except requests.exceptions.RequestException as e:
            return {
                "answer": f"⚠️ LLM request failed: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "none",
                "mode": "llm"
            }

    def check_health(self):
        """
        Check if Ollama is running and the configured model is available.

        Returns:
            dict with "status" ("ok" | "error"), "ollama_running", "model_available",
                  "available_models", and optional "message".
        """
        result = {
            "status": "error",
            "ollama_running": False,
            "model_available": False,
            "available_models": [],
            "current_model": self.model,
        }

        try:
            # Check if Ollama server is reachable
            response = requests.get(OLLAMA_API_TAGS, timeout=5)
            response.raise_for_status()
            data = response.json()

            result["ollama_running"] = True

            # List available models
            models = [m.get("name", "") for m in data.get("models", [])]
            result["available_models"] = models

            # Check if the configured model is available
            # Ollama model names can include tag suffixes like ":latest"
            model_found = any(
                m == self.model or m.startswith(f"{self.model}:")
                for m in models
            )
            result["model_available"] = model_found

            if model_found:
                result["status"] = "ok"
                result["message"] = f"Ollama is running with model '{self.model}' available."
            else:
                result["message"] = (
                    f"Ollama is running but model '{self.model}' is not installed. "
                    f"Available models: {', '.join(models) if models else 'none'}. "
                    f"Run `ollama pull {self.model}` to install it."
                )

        except requests.exceptions.ConnectionError:
            result["message"] = (
                "Ollama is not running. Install from https://ollama.com "
                "and run `ollama serve`."
            )

        except requests.exceptions.RequestException as e:
            result["message"] = f"Error checking Ollama: {str(e)}"

        return result

    def list_models(self):
        """
        List all models available in the local Ollama installation.

        Returns:
            dict with "models" list and "status".
        """
        try:
            response = requests.get(OLLAMA_API_TAGS, timeout=5)
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", "unknown"),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                })

            return {"models": models, "status": "ok"}

        except requests.exceptions.ConnectionError:
            return {
                "models": [],
                "status": "error",
                "message": "Ollama is not running."
            }

        except requests.exceptions.RequestException as e:
            return {
                "models": [],
                "status": "error",
                "message": str(e)
            }
