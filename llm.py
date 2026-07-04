"""Groq LLM istemcisi + kural tabanlı fallback."""
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

import config

load_dotenv()

_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        # .env'de yorum satırı birleşmiş olabilir — temizle
        key = key.split("#")[0].strip()
        _groq_client = Groq(api_key=key)
    return _groq_client


def _active_provider() -> Optional[str]:
    if config.LLM_PROVIDER == "none":
        return None
    if _get_groq():
        return "groq"
    return None


def llm_generate(system: str, user: str, json_mode: bool = False) -> Optional[str]:
    """LLM çağrısı. API yoksa None döner."""
    if not _active_provider():
        return None

    try:
        client = _get_groq()

        # SİNSİ GROQ HATASINA KARŞI ÖNLEM:
        # json_mode aktifse, prompt içinde mutlaka "JSON" kelimesi geçmeli.
        if json_mode and "json" not in system.lower() and "json" not in user.lower():
            system += "\nOutput must be a valid JSON object."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {
            "model": config.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[WARN] Groq LLM hatasi: {e}")
    return None


def parse_json_response(text: str) -> Optional[dict]:
    """LLM'den gelen metni güvenli bir şekilde JSON sözlüğüne çevirir."""
    if not text:
        return None

    # Markdown kod bloklarını (```json ... ```) temizle
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # JSON nesnesini metin içinden yakalamayı dene
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None