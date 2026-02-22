"""
GPT-4o zero-shot baseline. API key is read from the OPENAI_API_KEY environment variable; no key is stored in the repository. One request per call.
"""
import os
from typing import Any


def generate(
    prompt: str,
    model_name: str = "gpt-4o",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set. Do not commit keys.")
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    text = (response.choices[0].message.content or "").strip()
    return _extract_single_sentence(text)


def _extract_single_sentence(text: str) -> str:
    for sep in ["\n", ". ", ".\n"]:
        if sep in text:
            part = text.split(sep)[0].strip()
            if sep in (". ", ".\n") and not part.endswith("."):
                part = part + "."
            return part
    return text.strip()
