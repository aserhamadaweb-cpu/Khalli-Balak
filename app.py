"""
Khalli Balak — Gradio demo.

Loads Model 3 (multilingual embeddings + intent features + Logistic Regression)
and shows SCAM/SAFE/NOT SURE, confidence %, and plain-language WHY reasons.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import gradio as gr
import numpy as np

from src.explain import load_spam_corpus, most_similar_spam
from src.intent_features import extract_intent
from src.models import IntentEmbeddingLogRegModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model3_intent.joblib"
SPAM_CORPUS_PATH = ROOT / "models" / "spam_corpus.json"
NOT_SURE_THRESHOLD = 0.75

DISCLAIMER = (
    "This project is a prototype and is not financial advice. "
    "Even a confident result can be wrong. Do not send a PIN, OTP, or money because of this tool."
)
GITHUB_REPO = "https://github.com/aserhamadaweb-cpu/Khalli-Balak"
CREDIT_HTML = f"""
<div class="kb-credit">
  <span>Powered by <strong>ASER HAMADA</strong></span>
  <a href="{html.escape(GITHUB_REPO)}" target="_blank" rel="noopener noreferrer">GitHub repo — view the code</a>
</div>
"""

_model: IntentEmbeddingLogRegModel | None = None
_spam_texts: list[str] = []
_spam_embeddings: np.ndarray | None = None

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_FRANCO = re.compile(r"[A-Za-z]*[237][A-Za-z0-9]+|[A-Za-z]+[237]")
_CLUES: list[tuple[str, re.Pattern[str]]] = [
    ("Link", re.compile(r"https?://\S+|www\.\S+", re.I)),
    ("PIN", re.compile(r"\bpin\b|الرقم السري|رقم سري", re.I)),
    ("OTP or code", re.compile(r"\botp\b|كود|verification code", re.I)),
    ("Prize", re.compile(r"\bwon\b|\bprize\b|مبروك|جائزة|kesebt|كسبت", re.I)),
    ("Hurry", re.compile(r"urgent|immediately|عاجل|فوراً|فورا|delwa2ty|3agel", re.I)),
    ("Asks you to send", re.compile(r"\bsend\b|transfer|ابعت|حوّل|حول|ادفع|eb3at|edfa3", re.I)),
    ("WhatsApp", re.compile(r"whatsapp|واتساب", re.I)),
    ("Wallet or bank", re.compile(r"vodafone|instapay|فودافون|انستاباي|orange cash|فوري", re.I)),
]


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

    print("Loading Model 3 (embeddings + intent)...", flush=True)
    _model = IntentEmbeddingLogRegModel.load(MODEL_PATH)
    _model._ensure_encoder()

    print("Loading spam corpus for explanations...", flush=True)
    _spam_texts = load_spam_corpus(SPAM_CORPUS_PATH)
    # Small set keeps CPU demo startup fast.
    if len(_spam_texts) > 40:
        _spam_texts = _spam_texts[:40]
    print(f"Encoding {len(_spam_texts)} spam examples...", flush=True)
    _spam_embeddings = _model.encode(_spam_texts, batch_size=8)
    print("Ready.", flush=True)


def _card(kind: str, kicker: str, title: str, detail: str, pct: float) -> str:
    width = max(0, min(100, int(round(pct))))
    return f"""
    <article class="kb-card kb-{kind}">
      <p class="kb-kicker">{kicker}</p>
      <h2>{title}</h2>
      <p class="kb-detail">{detail}</p>
      <div class="kb-meter" aria-hidden="true"><span style="width:{width}%"></span></div>
      <p class="kb-pct">{pct:.1f}% confidence</p>
    </article>
    """


def guess_script(text: str) -> str:
    arabic = len(_ARABIC.findall(text))
    latin = len(_LATIN.findall(text))
    franco = len(_FRANCO.findall(text))
    if arabic >= 4 and arabic >= latin:
        return "Arabic"
    if franco >= 1 and latin >= arabic:
        return "Franco-Arabic"
    if latin >= arabic and latin > 0:
        return "English"
    if arabic:
        return "Arabic"
    return "Mixed"


def find_clues(text: str) -> tuple[list[str], str]:
    """Return clue labels and the message with those spans marked."""
    hits: list[tuple[int, int, str]] = []
    for label, pattern in _CLUES:
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), label))
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    labels: list[str] = []
    for start, end, label in hits:
        if start < last_end:
            continue
        chosen.append((start, end, label))
        last_end = end
        if label not in labels:
            labels.append(label)
    parts: list[str] = []
    cursor = 0
    for start, end, _label in chosen:
        parts.append(html.escape(text[cursor:start]))
        parts.append(f"<mark>{html.escape(text[start:end])}</mark>")
        cursor = end
    parts.append(html.escape(text[cursor:]))
    shown = "".join(parts) if text else ""
    return labels, shown


def _advice(kind: str) -> str:
    if kind == "scam":
        title = "What to do"
        lines = [
            "Do not tap the link.",
            "Do not send a PIN, OTP, or money.",
            "Open the official app yourself and check there.",
            "متبعتش رقم سري ولا فلوس من الرسالة.",
        ]
    elif kind == "unsure":
        title = "What to do"
        lines = [
            "Treat this as suspicious until you check.",
            "Open the official wallet or bank app.",
            "متعملش حاجة لحد ما تتأكد من التطبيق الرسمي.",
        ]
    else:
        title = "Still be careful"
        lines = [
            "A real bank does not ask for your PIN by text.",
            "If money moved, confirm it inside the official app.",
            "متشاركش الرقم السري حتى لو الرسالة شكلها عادية.",
        ]
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return f'<article class="kb-advice"><h3>{title}</h3><ul>{items}</ul></article>'


def _why_card(reasons: list[str], kind: str) -> str:
    items = "".join(f"<li>{html.escape(reason)}</li>" for reason in reasons)
    title = "Why this looks risky" if kind == "scam" else "Why this looks normal"
    if kind == "unsure":
        title = "Why we are not sure"
    return f"""
    <article class="kb-why">
      <p class="kb-kicker">Why this result</p>
      <h3>{title}</h3>
      <ul class="kb-why-list">{items}</ul>
    </article>
    """


def _neighbor_card(neighbor: str, score: float) -> str:
    safe_text = html.escape(neighbor)
    return f"""
    <article class="kb-why">
      <p class="kb-kicker">Extra context</p>
      <h3>Closest known scam</h3>
      <p class="kb-quote">“{safe_text}”</p>
      <p class="kb-sim">Similarity {score:.0%}</p>
    </article>
    """


def check_message(message: str) -> tuple[str, str]:
    if _model is None or _spam_embeddings is None:
        load_artifacts()

    text = (message or "").strip()
    if not text:
        empty = """
        <article class="kb-card kb-empty">
          <p class="kb-kicker">Waiting</p>
          <h2>Paste a message first</h2>
          <p class="kb-detail">Arabic, Franco-Arabic, or English.</p>
        </article>
        """
        return empty, ""

    result = _model.predict([text])
    prob = float(result.probabilities[0])
    is_scam = prob >= 0.5
    # Confidence of the chosen label. Below 75% we refuse a hard call.
    confidence = prob if is_scam else (1.0 - prob)
    intent = extract_intent(text)

    if confidence < NOT_SURE_THRESHOLD:
        lean = "SCAM" if is_scam else "SAFE"
        verdict = _card(
            "unsure",
            "Check again",
            "Not sure",
            f"Only {confidence * 100:.1f}% sure, and it leans {lean}. Open the official app before you act.",
            confidence * 100,
        )
        kind = "unsure"
    elif is_scam:
        verdict = _card(
            "scam",
            "Watch out",
            "Scam",
            "This looks like a scam text. Do not send a PIN, OTP, or money.",
            prob * 100,
        )
        kind = "scam"
    else:
        verdict = _card(
            "safe",
            "Looks normal",
            "Safe",
            "This looks like an ordinary message. Still confirm inside the official app.",
            confidence * 100,
        )
        kind = "safe"

    script = guess_script(text)
    labels, highlighted = find_clues(text)
    chips = "".join(f'<span class="kb-chip">{html.escape(label)}</span>' for label in labels)
    if not chips:
        chips = '<span class="kb-chip kb-chip-quiet">No obvious scam phrases</span>'

    why = _why_card(intent.reasons, kind)
    neighbor_html = ""
    if kind == "scam":
        query_vec = _model.encode([text])[0]
        neighbor, score = most_similar_spam(query_vec, _spam_embeddings, _spam_texts)
        neighbor_html = _neighbor_card(neighbor, score)
    details = f"""
    <section class="kb-extra">
      <p class="kb-script">Script guess: <strong>{html.escape(script)}</strong></p>
      {why}
      <article class="kb-marked" dir="auto"><p class="kb-kicker">Phrases we noticed</p>{chips}<p class="kb-preview">{highlighted}</p></article>
      {_advice(kind)}
      {neighbor_html}
    </section>
    """
    return verdict, details


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,680&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

.gradio-container {
  max-width: 860px !important;
  margin: 0 auto !important;
  padding: 28px 16px 48px !important;
  background:
    radial-gradient(820px 360px at 0% -8%, #f0e2c4 0%, transparent 58%),
    radial-gradient(680px 340px at 100% 0%, #d5ebe3 0%, transparent 52%),
    #f3eee4 !important;
}
.kb-hero {
  background:
    radial-gradient(420px 180px at 100% 0%, rgba(212, 162, 74, 0.18), transparent 60%),
    linear-gradient(155deg, #0e2a28 0%, #143f3b 58%, #1b564d 100%);
  color: #f7f1e6;
  border-radius: 30px;
  padding: 26px 26px 24px;
  box-shadow: 0 22px 46px rgba(14, 42, 40, 0.22);
  margin-bottom: 18px;
  border: 1px solid rgba(212, 162, 74, 0.28);
}
.kb-brand { display: flex; align-items: center; gap: 16px; }
.kb-logo {
  width: 64px;
  height: 64px;
  flex: 0 0 64px;
  filter: drop-shadow(0 8px 14px rgba(0, 0, 0, 0.18));
}
.kb-mark {
  display: inline-block;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 12px;
  color: #d4a24a;
  margin: 0 0 6px;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
}
.kb-hero h1 {
  font-family: Fraunces, "IBM Plex Sans Arabic", Georgia, serif;
  font-size: clamp(40px, 7vw, 64px);
  line-height: 0.95;
  margin: 0;
  font-weight: 560;
}
.kb-ar {
  display: block;
  margin-top: 8px;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
  font-size: clamp(22px, 4vw, 30px);
  font-weight: 600;
}
.kb-hero p {
  margin: 14px 0 0;
  max-width: 38rem;
  color: #efe7d8;
  font-size: 16px;
  line-height: 1.5;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
}
.kb-panel, .kb-side {
  background: rgba(255, 250, 243, 0.92) !important;
  border: 1px solid #e4d8c4 !important;
  border-radius: 24px !important;
  padding: 8px 8px 16px !important;
  box-shadow: 0 10px 30px rgba(70, 48, 20, 0.05);
}
label, .kb-panel span, textarea, button {
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif !important;
}
textarea {
  border-radius: 16px !important;
  background: #fffdf9 !important;
  border: 1px solid #e7dcc8 !important;
  font-size: 16px !important;
  line-height: 1.55 !important;
}
.kb-check {
  background: linear-gradient(180deg, #1a4e48 0%, #102c2a 100%) !important;
  color: #f7f1e6 !important;
  border: 1px solid #d4a24a !important;
  border-radius: 999px !important;
  min-height: 54px !important;
  font-size: 17px !important;
  font-weight: 650 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 10px 20px rgba(16, 44, 42, 0.16);
}
.kb-check:hover { background: #1b564d !important; }
.kb-card, .kb-why {
  border-radius: 22px;
  padding: 20px 20px 16px;
  margin-top: 8px;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
}
.kb-card h2, .kb-why h3 {
  font-family: Fraunces, "IBM Plex Sans Arabic", Georgia, serif;
  margin: 0 0 8px;
  font-size: 40px;
  line-height: 1;
  font-weight: 560;
}
.kb-why h3 { font-size: 26px; }
.kb-why-list { margin: 8px 0 0; padding-left: 1.15rem; line-height: 1.45; color: #2a3538; }
.kb-why-list li { margin: 0.25rem 0; }
.kb-kicker {
  margin: 0 0 6px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 12px;
  font-weight: 700;
}
.kb-detail, .kb-quote, .kb-sim, .kb-pct { margin: 0; line-height: 1.5; }
.kb-pct { margin-top: 8px; font-weight: 650; }
.kb-meter {
  height: 8px;
  border-radius: 99px;
  background: rgba(0, 0, 0, 0.08);
  margin-top: 14px;
  overflow: hidden;
}
.kb-meter span { display: block; height: 100%; border-radius: 99px; }
.kb-scam { background: #f8e6e1; color: #8c2e32; }
.kb-scam .kb-meter span { background: #c4514a; }
.kb-safe { background: #e4f4eb; color: #145c38; }
.kb-safe .kb-meter span { background: #1d8a57; }
.kb-unsure { background: #f8efd2; color: #7a5310; }
.kb-unsure .kb-meter span { background: #c9922a; }
.kb-empty { background: #f3eee4; color: #3d4a46; }
.kb-why { background: #fff; border: 1px solid #eadfcd; color: #24302c; }
.kb-quote {
  font-size: 15px;
  background: #faf6ef;
  border-radius: 14px;
  padding: 12px 14px;
  margin-top: 8px;
}
.kb-sim { margin-top: 8px; color: #6d6254; font-size: 13px; }
.kb-pills { display: flex; gap: 10px; margin-top: 12px; }
.kb-pill {
  flex: 1;
  background: #fff;
  border: 1px solid #eadfcd;
  border-radius: 16px;
  padding: 12px 14px;
}
.kb-pill span {
  display: block;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6d6254 !important;
}
.kb-pill strong {
  display: block;
  font-family: Fraunces, Georgia, serif;
  font-size: 28px;
  line-height: 1.1;
  margin-top: 4px;
}
.kb-pill em { font-style: normal; font-size: 13px; color: #5e564c; }
.kb-pill-scam strong { color: #9f2d2d; }
.kb-pill-safe strong { color: #146b3a; }
.kb-script { margin: 10px 2px 0; color: #5e564c; font-size: 13px; }
.kb-marked, .kb-advice {
  margin-top: 10px;
  background: #fff;
  border: 1px solid #eadfcd;
  border-radius: 18px;
  padding: 14px 16px;
  color: #24302c;
}
.kb-chip {
  display: inline-block;
  margin: 0 6px 6px 0;
  padding: 4px 10px;
  border-radius: 999px;
  background: #143f3b;
  color: #f7f1e6 !important;
  font-size: 12px;
}
.kb-chip-quiet { background: #efe6d6; color: #5e564c !important; }
.kb-preview { margin: 8px 0 0; line-height: 1.6; }
.kb-preview mark {
  background: #f6d36b;
  color: #1c2430;
  border-radius: 4px;
  padding: 0 3px;
}
.kb-advice h3 {
  margin: 0 0 6px;
  font-family: Fraunces, Georgia, serif;
  font-size: 22px;
}
.kb-advice ul { margin: 0; padding-left: 18px; }
.kb-advice li { margin: 4px 0; }
.kb-foot {
  margin-top: 18px;
  color: #5e564c;
  font-size: 13px;
  line-height: 1.45;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
}
.kb-credit {
  margin-top: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  align-items: center;
  justify-content: space-between;
  color: #3d4a48;
  font-size: 14px;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", sans-serif;
}
.kb-credit strong { color: #143f3b; letter-spacing: 0.04em; }
.kb-credit a {
  color: #143f3b;
  font-weight: 700;
  text-decoration: none;
  border-bottom: 1px solid #d4a24a;
}
@media (max-width: 700px) {
  .kb-hero { padding: 22px 18px; border-radius: 22px; }
  .kb-logo { width: 52px; height: 52px; flex-basis: 52px; }
  .kb-card h2 { font-size: 34px; }
  .kb-pills { flex-direction: column; }
}

/* Gradio follows the system dark theme. Keep this page in the light palette. */
html, body, body.dark {
  background: #f6f1e8 !important;
  color: #1c2430 !important;
  color-scheme: light !important;
}
.dark .form,
.form,
.dark .block.padded,
.block.padded,
.dark .wrap,
.wrap.default {
  background: #fffaf3 !important;
  color: #1c2430 !important;
  border-radius: 18px !important;
  border: 1px solid #eadfcd !important;
}
textarea, input, .dark textarea, button {
  color: #1c2430 !important;
  caret-color: #143f3b !important;
  unicode-bidi: plaintext;
  text-align: start;
}
textarea::placeholder { color: #8a7d6b !important; }
.kb-panel label, .kb-panel label span {
  color: #143f3b !important;
  background: transparent !important;
}
.kb-hero, .kb-hero h1, .kb-hero p, .kb-hero .kb-ar { color: #f7f1e6 !important; }
.kb-hero .kb-mark { color: #d4a24a !important; }
button.kb-check, button.kb-check span { color: #f8f3ea !important; }
.examples-table, .gallery, table.examples {
  background: transparent !important;
}

@keyframes kb-in {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: none; }
}
@keyframes kb-logo-in {
  from { opacity: 0; transform: scale(0.7); }
  to { opacity: 1; transform: none; }
}
@keyframes kb-splash-out {
  to { opacity: 0; visibility: hidden; }
}
.kb-splash {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: radial-gradient(420px 220px at 50% 42%, rgba(212, 162, 74, 0.16), transparent 62%), #0e2a28;
  animation: kb-splash-out 0.55s ease 1.2s forwards;
}
.kb-splash .kb-logo {
  width: 96px;
  height: 96px;
  flex-basis: 96px;
  animation: kb-logo-in 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.kb-splash h1 {
  margin: 0;
  color: #f7f1e6 !important;
  font-family: Fraunces, Georgia, serif;
  font-size: 44px;
  font-weight: 560;
  animation: kb-in 0.55s ease 0.18s both;
}
.kb-splash p {
  margin: 0;
  color: #d4a24a !important;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  font-size: 12px;
  animation: kb-in 0.55s ease 0.28s both;
}
.kb-hero { animation: kb-in 0.7s ease 0.2s both; }
.kb-panel { animation: kb-in 0.75s ease 0.35s both; }
.kb-card, .kb-extra, .kb-why, .kb-marked, .kb-advice { animation: kb-in 0.45s ease both; }
.kb-check { transition: transform 0.2s ease, background 0.2s ease; }
.kb-check:hover { transform: translateY(-1px); }
@media (prefers-reduced-motion: reduce) {
  .kb-splash { display: none; }
  .kb-hero, .kb-panel, .kb-card, .kb-extra, .kb-why, .kb-marked, .kb-advice, .kb-check {
    animation: none;
    transition: none;
  }
}
"""

LOGO = """
<svg class="kb-logo" viewBox="0 0 64 64" role="img" aria-label="Khalli Balak logo">
  <rect width="64" height="64" rx="18" fill="#d4a24a"/>
  <path d="M10 33c6.5-10 14-15 22-15s15.5 5 22 15c-6.5 10-14 15-22 15S16.5 43 10 33z" fill="none" stroke="#102c2a" stroke-width="3.2"/>
  <circle cx="32" cy="33" r="7.2" fill="#102c2a"/>
  <circle cx="34.2" cy="30.6" r="2.1" fill="#f7f1e6"/>
</svg>
"""

HERO = f"""
<section class="kb-hero">
  <div class="kb-brand">
    {LOGO}
    <div>
      <p class="kb-mark">Watch the message</p>
      <h1>Khalli Balak<span class="kb-ar">خلي بالك</span></h1>
    </div>
  </div>
  <p>Paste a text from anywhere. You get a verdict, the risky phrases, and what to do next.</p>
</section>
"""


THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "Segoe UI", "sans-serif"],
)


SPLASH = f"""
<div class="kb-splash" aria-hidden="true">
  {LOGO}
  <h1>Khalli Balak</h1>
  <p>خلي بالك</p>
</div>
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Khalli Balak — SMS Scam Check") as demo:
        gr.HTML(SPLASH)
        gr.HTML(HERO)
        with gr.Column(elem_classes=["kb-panel"]):
            inp = gr.Textbox(
                label="Message",
                lines=5,
                placeholder="الصق الرسالة هنا — Arabic, Franco, or English",
            )
            btn = gr.Button("Check this message", elem_classes=["kb-check"])
            verdict = gr.HTML()
            similar = gr.HTML()
        gr.Examples(
            label="Try one",
            examples=[
                ["Chase: we locked your card after a strange purchase. Enter the PIN at http://chase-unlock.example"],
                ["Your OTP is 4821 for Vodafone Cash login. Do not share it with anyone."],
                ["You received 500 EGP via InstaPay from Ahmed."],
                ["Congratulations, you are today's winner. Send 5 USD to receive the 1000 USD prize"],
                ["Mama, I paid the 200 for the lesson. See you later."],
                ["Your Vodafone Cash balance is 1,200 EGP."],
                ["Transaction of 350 EGP at Carrefour approved."],
                ["I'll be home after football. Don't wait for dinner"],
                ["فودافون كاش: حسابك هيتحظر. ابعت الـ PIN على http://vf-cash-secure.example"],
                ["انستاباي: تم تحويل 500 ج إلى محمد بنجاح. رقم العملية 998877"],
                ["رمز التحقق 8460. متشاركوش مع أي حد."],
                ["Eb3atly el OTP elly wesel delwa2ty 3ashan afata7 el wallet."],
            ],
            inputs=inp,
        )
        gr.HTML(CREDIT_HTML)
        gr.HTML(f'<p class="kb-foot">{html.escape(DISCLAIMER)}</p>')

        btn.click(
            fn=check_message,
            inputs=inp,
            outputs=[verdict, similar],
            show_progress="full",
        )
        inp.submit(
            fn=check_message,
            inputs=inp,
            outputs=[verdict, similar],
            show_progress="full",
        )
    return demo


if __name__ == "__main__":
    load_artifacts()
    ui = build_ui()
    # Set share=True if you need a temporary public link for judges.
    # (share tunnels can hang on some networks — local URL always works.)
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CUSTOM_CSS,
        footer_links=[],
        js="""() => {
          const dropDark = () => document.body && document.body.classList.remove("dark");
          dropDark();
          new MutationObserver(dropDark).observe(document.body, { attributes: true, attributeFilter: ["class"] });
        }""",
    )
