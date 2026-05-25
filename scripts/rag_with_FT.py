# Only switch to this AFTER fine-tuning
from __future__ import annotations

import os
from pathlib import Path

from mlx_lm import load, generate
from .retrieval import retrieve_info_v2

_MODEL = None
_TOKENIZER = None


def _resolve_adapter_path() -> str:
    """Resolve adapter path to an existing directory.

    Defaults to ADAPTER_PATH env var or ./adapters. If that doesn't exist,
    falls back to ./adapters_smoke (created by the smoke-test training).
    """

    project_root = Path(__file__).resolve().parent.parent

    configured = os.getenv("ADAPTER_PATH", "adapters")
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = project_root / configured_path

    if configured_path.exists():
        return str(configured_path)

    fallback = project_root / "adapters_smoke"
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        "Adapter path not found. Set ADAPTER_PATH to your trained adapter directory "
        "or create ./adapters by training with --adapter-path adapters. "
        f"Tried: {configured_path}"
    )


def _get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        adapter_path = _resolve_adapter_path()
        _MODEL, _TOKENIZER = load(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            adapter_path=adapter_path,
        )
    return _MODEL, _TOKENIZER

def generate_answer(user_question):
    model, tokenizer = _get_model()
    retrieved_docs = retrieve_info_v2(user_question, k=10)
    context_block = "\n".join(retrieved_docs)

    # Use the format you trained on
    prompt = f"<|im_start|>user\nUse the context to answer.\nContext: {context_block}\nQuestion: {user_question}<|im_end|>\n<|im_start|>assistant\n"

    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, 
        verbose=False
    )
    
    return response
