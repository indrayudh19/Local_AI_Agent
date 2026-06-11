"""
LLM Agents — specialized agent classes for the orchestrator.

ConversationAgent: Manages multi-turn conversation context and formatting.
ToolAgent:         Base class for future tool-use agents (summarize, translate, etc.).
"""

from llm.config import SYSTEM_PROMPT, MAX_HISTORY_TURNS


class ConversationAgent:
    """
    Manages conversation context for multi-turn LLM interactions.
    Handles system prompt injection, history trimming, and message formatting.
    """

    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def build_messages(self, user_message, history=None):
        """
        Build the full message list for the Ollama API.

        Includes system prompt, trimmed conversation history, and the
        current user message.

        Args:
            user_message: The current user message string.
            history:      Optional list of {"role": "user"|"assistant", "content": "..."}

        Returns:
            List of message dicts ready for the Ollama chat API.
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Append trimmed history
        if history:
            # Keep only the last MAX_HISTORY_TURNS * 2 messages (user + assistant pairs)
            trimmed = history[-(MAX_HISTORY_TURNS * 2):]
            for msg in trimmed:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Append the current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def format_response(self, raw_response):
        """
        Format the raw LLM response for the frontend.

        Args:
            raw_response: The raw text response from the LLM.

        Returns:
            dict with "answer" and metadata fields.
        """
        return {
            "answer": raw_response.strip(),
            "sources": [],
            "confidence": 1.0,
            "confidence_label": "none",
            "mode": "llm"
        }


class ToolAgent:
    """
    Base class for specialized tool agents.
    Future expansion: summarization, translation, code generation, etc.

    Each tool agent defines:
    - A system prompt tailored to the task
    - Input/output formatting specific to the tool
    """

    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt

    def build_messages(self, user_input):
        """
        Build messages for a single-turn tool invocation.

        Args:
            user_input: The input text for the tool.

        Returns:
            List of message dicts.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]


# ──────────────────────────────────────────────
# Pre-built Tool Agents (for future use)
# ──────────────────────────────────────────────

SUMMARIZER_AGENT = ToolAgent(
    name="summarizer",
    description="Summarizes long text into concise bullet points.",
    system_prompt=(
        "You are a text summarizer. Provide a concise summary "
        "of the given text in clear bullet points. "
        "Focus on the key ideas and main takeaways."
    )
)

EXPLAINER_AGENT = ToolAgent(
    name="explainer",
    description="Explains complex concepts in simple terms.",
    system_prompt=(
        "You are an expert explainer. Break down complex topics "
        "into simple, easy-to-understand language. "
        "Use analogies and examples where helpful."
    )
)
