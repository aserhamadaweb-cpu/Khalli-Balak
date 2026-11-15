"""Metrics, confusion matrices, wrong-prediction lists, and charts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ml_utils import confusion_counts, precision_recall_spam


def spam_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return precision_recall_spam(y_true, y_pred)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wrong_predictions(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    scripts: list[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for i, (t, yt, yp, p) in enumerate(
        zip(texts, y_true, y_pred, probabilities, strict=True)
    ):
        if int(yt) == int(yp):
            continue
        row = {
            "text": t,
            "true_label": int(yt),
            "pred_label": int(yp),
            "prob_spam": float(p),
        }
        if scripts is not None:
            row["script"] = scripts[i]
        rows.append(row)
    return rows


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_counts(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], ["SAFE (0)", "SCAM (1)"])
    ax.set_yticks([0, 1], ["SAFE (0)", "SCAM (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def high_confidence_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    min_confidence: float = 0.8,
) -> dict:
    """How often a confident call is actually correct.

    Confidence is the chance of the chosen label: max(p_spam, 1 - p_spam).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    mask = confidence >= min_confidence
    n = int(mask.sum())
    if n == 0:
        return {
            "min_confidence": min_confidence,
            "n_confident": 0,
            "n_correct": 0,
            "n_wrong": 0,
        }
    correct = int((y_true[mask] == y_pred[mask]).sum())
    return {
        "min_confidence": min_confidence,
        "n_confident": n,
        "n_correct": correct,
        "n_wrong": n - correct,
    }


def egypt_script_breakdown(
    scripts: list[str], y_true: np.ndarray, y_pred: np.ndarray
) -> list[dict]:
    rows: list[dict] = []
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    def one(script_name: str, mask: np.ndarray) -> dict:
        true_s = y_true[mask]
        pred_s = y_pred[mask]
        scam_mask = true_s == 1
        safe_mask = true_s == 0
        return {
            "script": script_name,
            "scams_total": int(scam_mask.sum()),
            "scams_caught": int(((pred_s == 1) & scam_mask).sum()),
            "safe_total": int(safe_mask.sum()),
            "false_alarms": int(((pred_s == 1) & safe_mask).sum()),
        }

    for script in sorted(set(scripts)):
        mask = np.array([s == script for s in scripts])
        rows.append(one(script, mask))
    rows.append(one("ALL", np.ones(len(scripts), dtype=bool)))
    return rows


def plot_egypt_by_script(breakdown: list[dict], out_path: Path, model_name: str) -> None:
    plot_rows = [r for r in breakdown if r["script"] != "ALL"]
    if not plot_rows:
        return
    labels = [r["script"] for r in plot_rows]
    caught = [r["scams_caught"] for r in plot_rows]
    false_alarms = [r["false_alarms"] for r in plot_rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, caught, width, label="Scams caught")
    ax.bar(x + width / 2, false_alarms, width, label="False alarms")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.set_title(f"Egypt test by script — {model_name}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
