"""Small ML helpers in NumPy/Torch (no scikit-learn — works under Windows Smart App Control)."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


def stratified_train_test_split(
    texts: list[str],
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Stratified 80/20 split with a fixed seed."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    x_train = [texts[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    return x_train, x_test, labels[train_idx], labels[test_idx]


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizerLite:
    """Simple TF-IDF with unigrams+bigrams (same idea as sklearn TfidfVectorizer)."""

    def __init__(self, max_features: int = 10000, min_df: int = 2) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _ngrams(self, tokens: list[str]) -> list[str]:
        grams = list(tokens)
        grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        return grams

    def fit(self, texts: list[str]) -> TfidfVectorizerLite:
        df_counter: Counter[str] = Counter()
        for text in texts:
            terms = set(self._ngrams(tokenize(text)))
            df_counter.update(terms)
        # Keep terms that appear in enough documents, then top by document frequency.
        candidates = [(t, c) for t, c in df_counter.items() if c >= self.min_df]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        chosen = candidates[: self.max_features]
        self.vocabulary_ = {term: i for i, (term, _) in enumerate(chosen)}
        n_docs = len(texts)
        df = np.zeros(len(self.vocabulary_), dtype=float)
        for term, idx in self.vocabulary_.items():
            df[idx] = df_counter[term]
        # Smooth IDF like sklearn default
        self.idf_ = np.log((n_docs + 1) / (df + 1)) + 1.0
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.idf_ is not None
        n = len(self.vocabulary_)
        X = np.zeros((len(texts), n), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = self._ngrams(tokenize(text))
            if not terms:
                continue
            counts = Counter(terms)
            for term, tf in counts.items():
                idx = self.vocabulary_.get(term)
                if idx is None:
                    continue
                X[row, idx] = tf
            # TF log scaling + IDF
            nonzero = X[row] > 0
            X[row, nonzero] = (1.0 + np.log(X[row, nonzero])) * self.idf_[nonzero]
            norm = np.linalg.norm(X[row])
            if norm > 0:
                X[row] /= norm
        return X

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class LogisticRegressionTorch:
    """Binary logistic regression with optional balanced class weights."""

    def __init__(
        self,
        class_weight: str | None = "balanced",
        lr: float = 0.05,
        epochs: int = 80,
        seed: int = 42,
    ) -> None:
        self.class_weight = class_weight
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.model: nn.Linear | None = None
        self.device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionTorch:
        torch.manual_seed(self.seed)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        n_features = X.shape[1]
        self.model = nn.Linear(n_features, 1).to(self.device)

        if self.class_weight == "balanced":
            n_pos = max(1, int((y == 1).sum()))
            n_neg = max(1, int((y == 0).sum()))
            # Higher weight for rarer class (spam).
            w_pos = (n_neg + n_pos) / (2.0 * n_pos)
            w_neg = (n_neg + n_pos) / (2.0 * n_neg)
            weights = torch.where(
                y_t > 0.5,
                torch.full_like(y_t, w_pos),
                torch.full_like(y_t, w_neg),
            )
        else:
            weights = torch.ones_like(y_t)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.model(X_t)
            loss = (loss_fn(logits, y_t) * weights).mean()
            loss.backward()
            opt.step()
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy().reshape(-1)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def precision_recall_spam(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "accuracy": float(accuracy),
        "precision_spam": float(precision),
        "recall_spam": float(recall),
        "f1_spam": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix for labels 0/1: rows=true, cols=pred."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        cm[t, p] += 1
    return cm
