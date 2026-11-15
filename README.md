# Khalli Balak — Scam texts in Egypt

**Question:** Does a scam-SMS filter trained on English still work for Egyptian Arabic and Franco-Arabic?

**This project is a prototype and is not financial advice.**

## For judges

- AI core: I trained my own classifiers. The demo does not ask ChatGPT if a text is a scam.
- English data is the public UCI SMS Spam Collection. The Egypt test file was never used in training.
- Comparison includes a baseline that always says SAFE.
- Numbers below are the real `train.py` output (seed 42). Weak spots are listed, not hidden.
- Demo: `python app.py`, then open the local URL. It loads `models/model2_local.joblib`.

## Problem

Egyptian parents and grandparents get SMS about Vodafone Cash, InstaPay, and banks. Some are real. Some are scams asking for a PIN, OTP, or a fake “verification” link. Most public spam datasets are English-only. This project checks whether an English-trained model transfers to local language styles.

## What I built

1. **Baseline** — always predict SAFE (never spam). Simple comparison.
2. **Model 1** — TF-IDF + Logistic Regression (`class_weight="balanced"`), English only.
3. **Model 2** — multilingual MiniLM embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) + Logistic Regression, English only.
4. **Model 2 local** — the same multilingual model, also trained on synthetic Egyptian and worldwide SMS that are **not** in the test file. This is what the demo loads.
5. A **Gradio web demo** that shows SCAM / SAFE / NOT SURE, a confidence bar, risky phrases, what to do next, and the most similar known English spam.
6. A **hand-written Egyptian test set** (`data/egypt_test.csv`) used **only for testing**.

## What I added (beyond the English dataset)

- Synthetic Egyptian Arabic, Franco-Arabic, and English wallet texts (`data/egypt_train_extra.csv`), kept separate from the test file.
- Synthetic worldwide scam and normal texts in English, French, Spanish, and German (`data/global_train_extra.csv`).
- Transfer evaluation by script (`arabic` / `franco` / `english`).
- A gold eye logo and a phone-friendly demo page.

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
           save models → Gradio demo loads model2_local
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
| model2_local | 0.758 | 0.946 |

Confusion matrices: `results/english_confusion_*.png`

### Egyptian test set (transfer)

| Model | Scams caught | False alarms |
|-------|--------------|--------------|
| baseline | **0 / 20** | **0 / 20** |
| model1_tfidf | **5 / 20** | **5 / 20** |
| model2_embed (English only) | **15 / 20** | **6 / 20** |
| model2_local (demo: English + Egypt + worldwide extras) | **19 / 20** | **1 / 20** |

On the Egypt test, when Model 2 local was at least **80%** confident, it was correct on **31 / 31** of those calls. The other 9 messages were less sure, and that is where the 2 mistakes sit. The demo therefore says **NOT SURE** below 80% instead of forcing SCAM or SAFE.

This is **not** a perfect model. The Egypt test has only 40 synthetic messages, written in a similar style to the extra training texts. A new real SMS can still be wrong. English spam precision is **0.758** for `model2_local` (recall **0.946**) because the model was balanced toward wallet and bank texts from Egypt and other countries.

By script for the demo model, `model2_local` (`results/egypt_metrics.csv`):

| Script | Scams caught | False alarms |
|--------|--------------|--------------|
| arabic | 7 / 8 | 1 / 6 |
| franco | 8 / 8 | 0 / 8 |
| english | 4 / 4 | 0 / 6 |

English-only Model 2 is weaker on the same file: Arabic 6/8 caught with 2 false alarms, Franco 5/8 with 1 false alarm, English 4/4 with 3 false alarms.

**Honest takeaway:** An English-only filter is not enough (TF-IDF catches 5/20 Egypt scams; multilingual embeddings catch 15/20 and still false-alarm). Training on extra Egyptian and worldwide texts, kept out of the test file, reaches 19/20 scams caught and 1/20 false alarms. That is better, and still not perfect. The remaining miss is one Arabic prize text.

Wrong predictions: `results/wrong_predictions_egypt_*.csv`

## Limitations

- Egypt test set is small (40 synthetic messages).
- Model 1 struggles when words are not English.
- The demo model still misses one Arabic prize scam and can flag one normal transfer receipt. Below 80% confidence it says NOT SURE.
- Not a bank product. Not financial advice. Do not rely on it to protect money.

## Next steps

- Collect a larger set of real public Egyptian SMS, not only synthetic examples.
- Test on messages written by someone else, so the style is not close to the training texts.

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
├── app.py                      # Gradio demo (model2_local)
├── train.py                    # Train + evaluate all models
├── requirements.txt
├── assets/logo.svg             # Eye logo
├── data/egypt_test.csv         # Egyptian test only — never trained on
├── data/egypt_train_extra.csv  # Extra Egyptian training texts
├── data/global_train_extra.csv # Extra worldwide training texts
├── models/                     # Saved classifiers, including model2_local
├── results/                    # Metrics, charts, wrong predictions
└── src/                        # Data loading, models, eval, explain
```

## Credits

- **Dataset:** UCI [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (same corpus as Hugging Face [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam))
- **Embedding model:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Libraries:** `numpy`, `torch`, `transformers`, `matplotlib`, `gradio`, `joblib`, `huggingface_hub`, `tqdm`
- **AI coding assistant:** Cursor (Composer) — helped write and organize this weekend project
- **Hackathon:** Veritas AI International High School Hackathon (AI + Finance track)

## Honesty note

Results are produced by `train.py` with seed `42` and an English stratified 80/20 split. The Egyptian file is never used in training. Failures and weak transfer are expected findings, not bugs to hide.
