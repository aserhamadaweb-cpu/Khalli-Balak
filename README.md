# Khalli Balak — Scam texts in Egypt

**Question:** Does a scam-SMS filter trained on English still work for Egyptian Arabic and Franco-Arabic?

**This project is a prototype and is not financial advice.**

## Problem

Egyptian parents and grandparents get SMS about Vodafone Cash, InstaPay, and banks. Some are real. Some are scams asking for a PIN, OTP, or a fake “verification” link. Most public spam datasets are English-only. This project checks whether an English-trained model transfers to local language styles.

## What I built

1. **Baseline** — always predict SAFE (never spam). Simple comparison.
2. **Model 1** — TF-IDF + Logistic Regression (`class_weight="balanced"`), English only.
3. **Model 2** — multilingual MiniLM embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) + Logistic Regression, English only.
4. A **Gradio web demo** that loads **Model 2**, shows SCAM/SAFE + confidence %, and the most similar known English spam as an explanation.
5. A **hand-written Egyptian test set** (`data/egypt_test.csv`) used **only for testing** (never for training).

## What I added (beyond the English dataset)

- Synthetic Egyptian Arabic / Franco-Arabic / English SMS about wallets and banks (no real private messages).
- Transfer evaluation by script (`arabic` / `franco` / `english`).
- Nearest-spam explanation in the demo.

## How it works (simple)

```text
English SMS (UCI / same as HF ucirvine/sms_spam)
        │
        ▼
  80/20 stratified split (seed=42)
        │
        ├─► train Baseline, Model 1, Model 2
        │
        ├─► evaluate on English holdout
        │
        └─► evaluate on Egypt test CSV (never seen in training)
                │
                ▼
           save Model 2 → Gradio demo
```

- **TF-IDF:** which English words/phrases appear (bag-of-words).
- **Embeddings:** turn a sentence into a meaning vector; the multilingual model can embed Arabic too.
- **Logistic Regression:** learns a line that separates scam vs safe from those features (balanced class weights because spam is rarer).
- **Baseline:** always SAFE — if a model cannot beat this on Egypt data, transfer failed.

> **Implementation note:** Logistic regression and TF-IDF are implemented with NumPy/Torch (same ideas as scikit-learn). On this Windows machine, Smart App Control blocked `scipy`/`sklearn` DLLs, so the project uses a dependency set that still trains a real classifier you own.

## Results (real run, seed=42)

English: **4,460 train / 1,114 test** messages. Egypt test: **40** messages (**20 scam / 20 safe**), never used in `fit()`.

### English held-out test (spam class)

| Model | Precision (spam) | Recall (spam) |
|-------|------------------|---------------|
| baseline | 0.000 | 0.000 |
| model1_tfidf | **0.939** | **0.933** |
| model2_embed | 0.874 | **0.980** |

Confusion matrices: `results/english_confusion_*.png`

### Egyptian test set (transfer)

| Model | Scams caught | False alarms |
|-------|--------------|--------------|
| baseline | **0 / 20** | **0 / 20** |
| model1_tfidf | **5 / 20** | **5 / 20** |
| model2_embed | **15 / 20** | **6 / 20** |

By script for Model 2 (`results/egypt_metrics.csv`):

| Script | Scams caught | False alarms |
|--------|--------------|--------------|
| arabic | 6 / 8 | 2 / 6 |
| franco | 5 / 8 | 1 / 8 |
| english | 4 / 4 | 3 / 6 |

**Honest takeaway:** English models work well on English SMS. On Egyptian Arabic/Franco, TF-IDF barely beats guessing. Multilingual embeddings transfer better (15/20 scams) but still miss some scams and raise false alarms — so an English-only filter is **not enough** for Egyptian parents without local data.

Wrong predictions: `results/wrong_predictions_egypt_*.csv`

## Limitations

- Egypt test set is small (40 synthetic messages).
- Model 1 struggles when words are not English.
- Model 2 still false-alarms on some normal texts.
- Not a bank product. Not financial advice. Do not rely on it to protect money.

## Next steps

- Add a small Arabic/Franco training set and re-measure carefully (stretch).
- Collect a larger community Egyptian SMS corpus.
- Tune the decision threshold to reduce false alarms for grandparents.

## How to run (fresh clone)

### 1) Requirements

- Python 3.10+ (3.12 recommended)
- Internet (first run downloads the SMS dataset + MiniLM model)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Train + evaluate

```bash
python train.py
```

Writes:

- `models/` — baseline, Model 1, Model 2, spam corpus
- `results/` — metrics tables, confusion matrices, wrong-prediction lists
- `data/sms_spam_english.csv` — cached English data after first download

### 3) Launch the demo

```bash
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

For a temporary public link for judges, set `share=True` in `app.py` → `ui.launch(...)`.

### Hosting suggestion

**Hugging Face Spaces** (Gradio) with `app.py`, `requirements.txt`, `src/`, and `models/`.  
Fallback: Gradio `share=True` link from your laptop during judging.

## Project layout

```text
├── app.py                 # Gradio demo (Model 2)
├── train.py               # Train + evaluate all models
├── requirements.txt
├── data/egypt_test.csv    # Egyptian test-only messages
├── models/                # Saved classifiers
├── results/               # Metrics + charts
└── src/                   # Data loading, models, eval, explain
```

## Credits

- **Dataset:** UCI [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (same corpus as Hugging Face [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam))
- **Embedding model:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Libraries:** `numpy`, `torch`, `transformers`, `matplotlib`, `gradio`, `joblib`, `huggingface_hub`, `tqdm`
- **AI coding assistant:** Cursor (Composer) — helped write and organize this weekend project
- **Hackathon:** Veritas AI International High School Hackathon (AI + Finance track)

## Honesty note

Results are produced by `train.py` with seed `42` and an English stratified 80/20 split. The Egyptian file is never used in training. Failures and weak transfer are expected findings, not bugs to hide.
