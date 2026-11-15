"""
Khalli Balak — Gradio demo.

Loads trained Model 2 (multilingual embeddings + Logistic Regression)
and shows SCAM/SAFE, confidence %, and the most similar known English spam.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from src.explain import load_spam_corpus, most_similar_spam
from src.models import EmbeddingLogRegModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model2_embed_lr.joblib"
SPAM_CORPUS_PATH = ROOT / "models" / "spam_corpus.json"

DISCLAIMER = (
    "This project is a prototype and is not financial advice. "
    "It was trained mainly on English SMS and may be wrong on Arabic or Franco-Arabic."
)

_model: EmbeddingLogRegModel | None = None
_spam_texts: list[str] = []
_spam_embeddings: np.ndarray | None = None


def load_artifacts() -> None:
    global _model, _spam_texts, _spam_embeddings
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run `python train.py` first to create models/."
        )
    if not SPAM_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPAM_CORPUS_PATH}. Run `python train.py` first."
        )

    print("Loading Model 2...")
    _model = EmbeddingLogRegModel.load(MODEL_PATH)
    _model._ensure_encoder()

    print("Loading spam corpus for explanations...")
    _spam_texts = load_spam_corpus(SPAM_CORPUS_PATH)
    # Cap neighbor search for demo speed (still representative).
    if len(_spam_texts) > 400:
        _spam_texts = _spam_texts[:400]
    _spam_embeddings = _model.encode(_spam_texts)
    print("Ready.")


def check_message(message: str) -> tuple[str, str, str]:
    if _model is None or _spam_embeddings is None:
        load_artifacts()

    text = (message or "").strip()
    if not text:
        return ("Please paste an SMS first.", "", DISCLAIMER)

    result = _model.predict([text])
    prob = float(result.probabilities[0])
    is_scam = int(result.labels[0]) == 1

    if is_scam:
        verdict = f"SCAM 🚨\nConfidence: {prob * 100:.1f}% chance this looks like spam/scam"
    else:
        safe_conf = (1.0 - prob) * 100
        verdict = f"SAFE ✅\nConfidence: {safe_conf:.1f}% chance this looks safe"

    query_vec = _model.encode([text])[0]
    neighbor, score = most_similar_spam(query_vec, _spam_embeddings, _spam_texts)
    explanation = (
        f"Most similar known English spam (cosine similarity {score:.2f}):\n\n"
        f"\"{neighbor}\""
    )
    return verdict, explanation, DISCLAIMER


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Khalli Balak — SMS Scam Check") as demo:
        gr.Markdown(
            """
# Khalli Balak
### Does an English-trained scam filter work for Egyptian Arabic & Franco-Arabic?

Paste an SMS (Arabic, Franco-Arabic, or English). The demo uses **your trained Model 2**
(multilingual embeddings + logistic regression) — not ChatGPT.
"""
        )
        with gr.Row():
            inp = gr.Textbox(
                label="Paste SMS here",
                lines=5,
                placeholder="مثال: فودافون كاش... / VF Cash: eb3at el PIN...",
            )
        btn = gr.Button("Check", variant="primary")
        verdict = gr.Textbox(label="Result", lines=3)
        similar = gr.Textbox(label="Why? (most similar known spam)", lines=6)
        disclaimer = gr.Textbox(label="Disclaimer", value=DISCLAIMER, lines=2)

        btn.click(fn=check_message, inputs=inp, outputs=[verdict, similar, disclaimer])
        inp.submit(fn=check_message, inputs=inp, outputs=[verdict, similar, disclaimer])

        gr.Examples(
            examples=[
                ["فودافون كاش: حسابك هيتحظر. ابعت الـ PIN على http://vf-cash-secure.example"],
                ["VF Cash: estalamt 80g men Omar. el balance el gedeed 520g"],
                ["Bank: Purchase of 75 EGP at Cairo Metro approved on your debit card"],
                ["InstaPay ALERT: Verify identity at http://insta-secure.example or transfer cancels"],
            ],
            inputs=inp,
        )
    return demo


if __name__ == "__main__":
    load_artifacts()
    ui = build_ui()
    # share=True gives a temporary public link for judges if Spaces is unavailable.
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
