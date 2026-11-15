"""Three classifiers: baseline, TF-IDF+LR, multilingual embeddings+LR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.ml_utils import LogisticRegressionTorch, TfidfVectorizerLite


@dataclass
class PredictResult:
    labels: np.ndarray
    probabilities: np.ndarray  # P(spam)


class AlwaysSafeBaseline:
    """Baseline that always predicts SAFE (not spam)."""

    name = "baseline"

    def fit(self, texts: list[str], labels: np.ndarray) -> AlwaysSafeBaseline:
        return self

    def predict(self, texts: list[str]) -> PredictResult:
        n = len(texts)
        return PredictResult(
            labels=np.zeros(n, dtype=int),
            probabilities=np.zeros(n, dtype=float),
        )

    def save(self, path) -> None:
        joblib.dump({"type": "baseline"}, path)

    @classmethod
    def load(cls, path) -> AlwaysSafeBaseline:
        joblib.load(path)
        return cls()


class TfidfLogRegModel:
    """Model 1: TF-IDF bag-of-words + logistic regression (English features)."""

    name = "model1_tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizerLite(max_features=10000, min_df=2)
        self.clf = LogisticRegressionTorch(class_weight="balanced", epochs=60)

    def fit(self, texts: list[str], labels: np.ndarray) -> TfidfLogRegModel:
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> PredictResult:
        X = self.vectorizer.transform(texts)
        proba = self.clf.predict_proba(X)
        return PredictResult(labels=(proba >= 0.5).astype(int), probabilities=proba)

    def save(self, path) -> None:
        joblib.dump({"type": "tfidf_lr", "vectorizer": self.vectorizer, "clf": self.clf}, path)

    @classmethod
    def load(cls, path) -> TfidfLogRegModel:
        payload = joblib.load(path)
        obj = cls()
        obj.vectorizer = payload["vectorizer"]
        obj.clf = payload["clf"]
        return obj


class EmbeddingLogRegModel:
    """Model 2: multilingual MiniLM embeddings + logistic regression.

    Encoder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    (loaded with Hugging Face Transformers + mean pooling).
    """

    name = "model2_embed"
    EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        self.tokenizer: Any | None = None
        self.encoder: Any | None = None
        self.clf = LogisticRegressionTorch(class_weight="balanced", epochs=80)
        self._fitted = False
        self.device = torch.device("cpu")

    def _ensure_encoder(self) -> None:
        if self.encoder is None:
            print(f"Loading embedding model: {self.EMBED_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.EMBED_MODEL_ID)
            self.encoder = AutoModel.from_pretrained(self.EMBED_MODEL_ID)
            self.encoder.to(self.device)
            self.encoder.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_encoder()
        assert self.tokenizer is not None and self.encoder is not None
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.encoder(**encoded)
                # Mean pooling over tokens (standard sentence-transformers style)
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (out.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vectors.append(emb.cpu().numpy())
        if not vectors:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack(vectors)

    def fit(self, texts: list[str], labels: np.ndarray) -> EmbeddingLogRegModel:
        X = self.encode(list(texts))
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> PredictResult:
        if not self._fitted:
            raise RuntimeError("Model 2 is not fitted yet.")
        X = self.encode(list(texts))
        proba = self.clf.predict_proba(X)
        return PredictResult(labels=(proba >= 0.5).astype(int), probabilities=proba)

    def save(self, path) -> None:
        joblib.dump(
            {
                "type": "embed_lr",
                "embed_model_id": self.EMBED_MODEL_ID,
                "clf": self.clf,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> EmbeddingLogRegModel:
        payload = joblib.load(path)
        obj = cls()
        obj.EMBED_MODEL_ID = payload["embed_model_id"]
        obj.clf = payload["clf"]
        obj._fitted = True
        return obj
