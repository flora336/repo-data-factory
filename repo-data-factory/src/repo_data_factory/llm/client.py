from __future__ import annotations
from dataclasses import dataclass
import os
import time
from typing import Any, Dict, Optional, Tuple
import json
import requests

@dataclass
class LLMConfig:
    """OpenAI-compatible config.

    IMPORTANT: requests will use HTTP(S)_PROXY from environment by default.
    If you are calling an internal gateway (e.g. 172.*), and your env sets a local proxy like 127.0.0.1:49091,
    you may see timeouts. We disable trust_env by default to avoid unexpected proxy routing.
    """
    base_url: str = os.getenv("RDF_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("RDF_LLM_API_KEY", "")
    model: str = os.getenv("RDF_LLM_MODEL", "gpt-4.1-mini")
    temperature: float = float(os.getenv("RDF_LLM_TEMPERATURE", "0.2"))
    timeout_s: int = int(os.getenv("RDF_LLM_TIMEOUT_S", "60"))
    max_output_tokens: int = int(os.getenv("RDF_LLM_MAX_TOKENS", "2048"))
    disable_env_proxy: bool = os.getenv("RDF_LLM_DISABLE_ENV_PROXY", "1") in {"1","true","True","yes","Y"}

class OpenAIChatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.session = requests.Session()
        if cfg.disable_env_proxy:
            # Critical: prevent requests from picking up HTTP_PROXY / HTTPS_PROXY
            self.session.trust_env = False

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.api_key:
            raise RuntimeError("Missing API key: set RDF_LLM_API_KEY")
        url = self.cfg.base_url.rstrip("/") + path
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        r = self.session.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()

    def chat_text(self, system: str, user: str) -> str:
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "max_tokens": self.cfg.max_output_tokens,
        }
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of the first JSON object from a possibly chatty response.

    Works without regex recursion (Python re does not support (?R)).
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    # Fast path: full JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Stack-based brace matching
    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in response")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i+1]
                try:
                    obj = json.loads(chunk)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON chunk: {e}. Chunk head: {chunk[:200]}")
                if not isinstance(obj, dict):
                    raise ValueError("Extracted JSON is not an object")
                return obj
    raise ValueError("Unclosed JSON object in response")
