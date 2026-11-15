# Data notes

## English training data

Downloaded at train time from the UCI Machine Learning Repository (same SMS Spam Collection published on Hugging Face as [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam)):

- Source URL used by `train.py`: UCI SMS Spam Collection zip
- Cached locally as `sms_spam_english.csv` after first download
- Labels: `0` = ham (safe), `1` = spam

## Egyptian test set (`egypt_test.csv`)

- Written for this hackathon as **synthetic** examples inspired by common mobile-wallet / bank SMS patterns in Egypt.
- Columns: `text`, `label` (`1` = scam, `0` = normal), `script` (`arabic` / `franco` / `english`)
- **TEST ONLY** — never used for training.
- No real phone numbers, no private messages, no personal data.
- Fake short codes and amounts are placeholders (e.g. `01200000000`, `*9*`).
