"""
Deep Research module using Gemini's search capabilities.
"""
from config import GEMINI_API_KEY
import json

# Try to import the new google-genai sdk, fallback to google.generativeai if needed
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    try:
        import google.generativeai as genai_legacy
        HAS_LEGACY = True
    except ImportError:
        HAS_LEGACY = False

class DeepResearchAgent:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.client = None
        
        if self.api_key:
            if HAS_GENAI:
                self.client = genai.Client(api_key=self.api_key)
            elif HAS_LEGACY:
                genai_legacy.configure(api_key=self.api_key)
    
    def research(self, query: str):
        """
        Perform deep research using Gemini with Google Search tool.
        """
        if not self.api_key:
            return {
                "answer": "Error: GEMINI_API_KEY is not configured. Please add your Gemini API key to the config.py file to enable Deep Research.",
                "sources": [],
                "confidence_label": "none"
            }
            
        try:
            # We attempt to use the new Google GenAI SDK first
            if HAS_GENAI and self.client:
                # Use Gemini 2.5 Pro with Google Search enabled
                response = self.client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=query,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        temperature=0.4
                    )
                )
                
                # Extract grounding metadata (sources) if available
                sources = []
                # Grounding metadata might be nested depending on the response format
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        gm = candidate.grounding_metadata
                        if hasattr(gm, 'grounding_chunks'):
                            for chunk in gm.grounding_chunks:
                                if hasattr(chunk, 'web') and chunk.web:
                                    sources.append({
                                        "file": chunk.web.title or "Web Source",
                                        "url": chunk.web.uri,
                                        "page": ""
                                    })
                
                return {
                    "answer": response.text,
                    "sources": sources,
                    "confidence_label": "high" if sources else "medium"
                }
                
            elif HAS_LEGACY:
                # Fallback to older google-generativeai package if that's what's installed
                # (Note: Google Search tool configuration might differ or not be fully supported in old versions)
                model = genai_legacy.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(query)
                return {
                    "answer": response.text,
                    "sources": [],
                    "confidence_label": "medium"
                }
                
            else:
                return {
                    "answer": "Error: google-genai package is not installed. Please install it using: pip install google-genai",
                    "sources": [],
                    "confidence_label": "none"
                }
                
        except Exception as e:
            return {
                "answer": f"An error occurred during deep research: {str(e)}",
                "sources": [],
                "confidence_label": "none"
            }
