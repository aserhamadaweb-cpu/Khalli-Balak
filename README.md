# Khalli Balak — Scam texts in Egypt

**Question:** Can we tell a scam SMS from a real money notice by *what it asks you to do*, not by money words alone?

**This project is a prototype and is not financial advice.**

## Live website

**Try it now:** [https://khalli-balak.vercel.app](https://khalli-balak.vercel.app)

Built by **ASER HAMADA**. Repo: [github.com/aserhamadaweb-cpu/Khalli-Balak](https://github.com/aserhamadaweb-cpu/Khalli-Balak)

## For judges

- **Demo model accuracy on Egypt test: 95%** (`model3_intent`, 38/40 correct, seed 42). Spam precision **100%**, SAFE money false positives **0**. Still misses 2 scams (shown as Not sure when confidence &lt; 75%).
- AI core: I trained my own classifiers. The demo does **not** ask ChatGPT/Gemini if a text is a scam.
- English data is the public UCI SMS Spam Collection. `egypt_test.csv` was **never** used in training.
- Comparison includes a baseline that always says SAFE, older text-only models, and the new intent model.
- Numbers below are the real `train.py` output. Weak spots are listed, not hidden.
- Local run: `python app.py` → loads `models/model3_intent.joblib`.

## Problem

Egyptian parents and grandparents get SMS about Vodafone Cash, InstaPay, and banks. Some are real notices. Some are scams that ask for a PIN, OTP, fee, or fake link. Older filters treat “money words” as suspicious. Being about money is not what makes a message a scam. **What matters is the ask.**

## What I built

1. **Baseline** — always predict SAFE (never spam).
2. **Model 1** — TF-IDF + Logistic Regression (`class_weight="balanced"`), English only.
3. **Model 2** — multilingual MiniLM embeddings + Logistic Regression, English only.
4. **Model 2 local** — same embeddings, also trained on synthetic Egyptian + worldwide SMS (not in the test file).
5. **Model 3 intent (demo)** — multilingual embeddings **plus intent features**, trained on UCI English SMS + self-written `hard_negatives_train.csv` (SAFE money notices and SCAM asks). Never trains on `egypt_test.csv`.
6. A **web demo** that shows SCAM / SAFE / NOT SURE, plain-language WHY reasons, and hides “closest known scam” when the verdict is SAFE.

## Intent features (what the message asks)

| Feature | Idea |
|---------|------|
| asks_to_share_code | Send/share OTP, PIN, password (EN / AR / Franco) |
| warns_not_to_share | “Do not share” / متشاركش — SAFE signal for real OTP notices |
| has_link / unofficial_link | Any link, and whether it is not an official domain |
| urgency_or_threat | Blocked, suspended, now, هيتحظر, 3agel… |
| prize_or_too_good | Won / gift / جايزة / mabrook… |
| asks_to_send_money | Pay a fee / send money (not completed receipts) |
| pure_notice | Received / approved / balance / تم استلام — no ask |

## Results (real run, seed=42)

English: **4,460 train / 1,114 test**. Egypt test: **40** messages (**20 scam / 20 safe**), never used in `fit()`.

### Old vs new on Egypt test only

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 | Confusion (TN/FP/FN/TP) |
|-------|----------|------------------|---------------|-----|-------------------------|
| baseline | 0.500 | 0.000 | 0.000 | 0.000 | 20 / 0 / 20 / 0 |
| model1_tfidf | 0.500 | 0.500 | 0.250 | 0.333 | 15 / 5 / 15 / 5 |
| model2_embed | 0.725 | 0.714 | 0.750 | 0.732 | 14 / 6 / 5 / 15 |
| model2_local (old demo) | **0.950** | 0.950 | **0.950** | **0.950** | 19 / 1 / 1 / 19 |
| **model3_intent (new demo)** | **0.950** | **1.000** | 0.900 | 0.947 | **20 / 0 / 2 / 18** |

Files: `results/egypt_compare_old_vs_new.csv`, `results/egypt_confusion_*.png`

### SAFE money false positives (the main weakness)

Count of **SAFE** Egypt-test messages that talk about money/banks but were flagged **SCAM**:

| Model | SAFE money false positives |
|-------|----------------------------|
| baseline | 0 |
| model1_tfidf | **5** |
| model2_embed | **5** |
| model2_local | **1** |
| **model3_intent** | **0** |

File: `results/egypt_safe_money_fp_summary.csv`

### English held-out spam (for reference)

| Model | Precision | Recall |
|-------|-----------|--------|
| baseline | 0.000 | 0.000 |
| model1_tfidf | 0.939 | 0.933 |
| model2_embed | 0.874 | 0.980 |
| model2_local | 0.758 | 0.946 |
| model3_intent | **0.910** | **0.953** |

### Honest takeaway

- Text-only models often treat money language as scam-like (many SAFE money false alarms).
- **Model 3 cuts SAFE money false positives on the Egypt test from 5 (English embed) / 1 (local) down to 0**, with spam precision **1.000**.
- It still **misses 2 / 20** Egypt scams (recall 0.900): one “call and confirm your PIN” bank lure, and one Franco “dad, car broke, send VF Cash” story. Both sit under 75% confidence, so the demo says **NOT SURE** instead of SAFE.
- Model 2 local still catches one more scam (19/20) but keeps one SAFE money false alarm. The demo ships Model 3 because the goal was intent, not keyword panic.
- When Model 3 was ≥75% confident on Egypt, it was correct on **38 / 38** calls.

Wrong predictions: `results/wrong_predictions_egypt_model3_intent.csv`

## Limitations

- Egypt test set is small (40 synthetic messages).
- Model 3 can still miss social-engineering stories that look like family chat, and “call this number and confirm your PIN” bank texts.
- Intent rules are hand-written keyword/pattern features (EN/AR/Franco), not a second neural net.
- Not a bank product. Not financial advice.

## Next steps

- Larger real public Egyptian SMS, written by someone else.
- Better handling of voice-call + confirm-PIN scams and family-emergency money asks.

## How to run (fresh clone)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py                 # all models
# or: python train.py --intent-only
python app.py                   # http://127.0.0.1:7860
```

Online: https://khalli-balak.vercel.app

## Project layout

```text
├── app.py                         # Demo (model3_intent)
├── api/index.py                   # Vercel FastAPI checker
├── public/index.html              # Public web page
├── train.py
├── data/egypt_test.csv            # TEST ONLY
├── data/hard_negatives_train.csv  # Self-written SAFE money + SCAM asks
├── data/egypt_train_extra.csv     # Used by model2_local only
├── data/global_train_extra.csv
├── models/model3_intent.joblib
├── results/egypt_compare_old_vs_new.csv
└── src/intent_features.py
```

## Credits

- **Dataset:** UCI [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (same corpus as Hugging Face [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam))
- **Hard negatives:** self-written for this hackathon (see `data/hard_negatives_train.csv`)
- **Embedding model:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Libraries:** `numpy`, `torch`, `transformers`, `matplotlib`, `gradio`, `fastapi`, `joblib`, `huggingface_hub`, `tqdm`
- **AI coding assistant:** Cursor (Composer) — helped write and organize this weekend project
- **Hackathon:** Veritas AI International High School Hackathon (AI + Finance track)

## Honesty note

Results are produced by `train.py` with seed `42`. `egypt_test.csv` is never used in training. Failures stay listed.
