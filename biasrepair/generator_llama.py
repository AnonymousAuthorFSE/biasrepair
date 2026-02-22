"""
LLaMA 3.1–8B-Instruct wrapper via transformers. Used for candidate generation with configurable temperature and sampling.
"""
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llama(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device_map: str | None = "auto", seed: int = 42):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    seed: int | None = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_single_sentence(decoded)


def _extract_single_sentence(text: str) -> str:
    """Take first sentence only; strip explanation."""
    text = text.strip()
    for sep in ["\n", ". ", ".\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()
            if sep == ". " or sep == ".\n":
                text = text + "." if not text.endswith(".") else text
            break
    return text.strip()
