"""Nearest known spam message — simple explanation for the demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_spam_corpus(texts: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"spam_texts": list(texts)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_spam_corpus(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["spam_texts"])


def most_similar_spam(
    query_embedding: np.ndarray,
    spam_embeddings: np.ndarray,
    spam_texts: list[str],
) -> tuple[str, float]:
    """Return the spam text with highest cosine similarity to the query.

    Embeddings are assumed L2-normalized, so cosine = dot product.
    """
    if len(spam_texts) == 0:
        return ("(no spam examples stored)", 0.0)
    scores = spam_embeddings @ query_embedding.reshape(-1)
    idx = int(np.argmax(scores))
    return spam_texts[idx], float(scores[idx])
