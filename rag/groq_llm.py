import json
from typing import List, Dict, Optional
from groq import Groq
import os

class GroqLLM:
    """
    Single responsibility:
      - Call Groq chat completions (stream or non-stream).
    """
    def __init__(self, model: str):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # uses GROQ_API_KEY env var
        self.model = model

    def chat(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        response_format: Optional[Dict] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=stream,
            response_format=response_format,
        )

        if stream:
            out = ""
            for chunk in resp:
                out += chunk.choices[0].delta.content or ""
            return out

        return resp.choices[0].message.content

    def chat_json(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
    ) -> Dict:
        """
        Forces JSON output. If Groq SDK/model supports response_format json_object,
        parsing becomes stable for checker verdict.
        """
        text = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format={"type": "json_object"},
        )
        return json.loads(text)
