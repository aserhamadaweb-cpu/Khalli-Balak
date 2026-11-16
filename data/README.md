# Data notes

## English training data

Downloaded at train time from the UCI Machine Learning Repository (same SMS Spam Collection published on Hugging Face as [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam)):

- Source URL used by `train.py`: UCI SMS Spam Collection zip
- Cached locally as `sms_spam_english.csv` after first download
- Labels: `0` = ham (safe), `1` = spam

## Worldwide training texts (`global_train_extra.csv`)

- Synthetic scam and normal SMS from outside Egypt (English, French, Spanish, German).
- Training only. Not part of the Egypt test file.
- Repeated with the same scam/safe weights as the Egyptian extras.

## Extra Egyptian training texts (`egypt_train_extra.csv`)

- Separate from the test file. `train.py` refuses to run if any sentence appears in both.
- Used only to train `model2_local`.
- Each scam text is repeated 12 times and each safe text 40 times so the small local set is not ignored next to 4,460 English SMS.
- Still synthetic. No real phone numbers.

## Hard negatives (`hard_negatives_train.csv`)

- Self-written SAFE money notices (~60) and SCAM asks (~30) in English, Egyptian Arabic, and Franco-Arabic.
- Goal: teach the model that money words alone are not a scam — the *ask* matters (share a code, pay a fee, tap a shady link).
- Used only to train `model3_intent` (with the UCI English set). Repeated so they are not drowned out.
- Checked for no overlap with `egypt_test.csv`. Still synthetic. No real phone numbers.

## Egyptian test set (`egypt_test.csv`)

- Written for this hackathon as **synthetic** examples inspired by common mobile-wallet / bank SMS patterns in Egypt.
- Columns: `text`, `label` (`1` = scam, `0` = normal), `script` (`arabic` / `franco` / `english`)
- **TEST ONLY** — never used for training.
- No real phone numbers, no private messages, no personal data.
- Fake short codes and amounts are placeholders (e.g. `01200000000`, `*9*`).
