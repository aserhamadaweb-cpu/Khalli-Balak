"""
Train and evaluate Khalli Balak scam-SMS classifiers.

Models (English train only):
  1) Baseline — always predict SAFE
  2) Model 1 — TF-IDF + Logistic Regression
  3) Model 2 — multilingual MiniLM embeddings + Logistic Regression

Egyptian CSV is TEST ONLY (never fitted on).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.data_loading import (
    RANDOM_SEED,
    english_split,
    load_egypt_test,
    load_egypt_train_extra,
    load_global_train_extra,
    load_english_sms,
)
from src.evaluate import (
    egypt_script_breakdown,
    high_confidence_stats,
    plot_confusion_matrix,
    plot_egypt_by_script,
    spam_precision_recall,
    write_csv,
    wrong_predictions,
)
from src.explain import save_spam_corpus
from src.models import AlwaysSafeBaseline, EmbeddingLogRegModel, TfidfLogRegModel

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
# Repeat local SMS so 4,460 English rows do not drown them out.
# Safe texts are repeated more often so normal bills and transfers stay SAFE.
LOCAL_REPEAT_SCAM = 12
LOCAL_REPEAT_SAFE = 40


def run(local_only: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading English SMS Spam Collection...")
    texts, labels = load_english_sms()
    x_train, x_test, y_train, y_test = english_split(texts, labels, seed=RANDOM_SEED)
    print(
        f"English split (seed={RANDOM_SEED}): "
        f"train={len(x_train)}, test={len(x_test)}, "
        f"spam_train={int(y_train.sum())}, spam_test={int(y_test.sum())}"
    )

    eg_texts, eg_labels, eg_scripts = load_egypt_test()
    print(
        f"Egypt test (held-out forever): n={len(eg_texts)}, "
        f"scams={int(eg_labels.sum())}, "
        f"scripts={sorted(set(eg_scripts))}"
    )

    spam_train_texts = [t for t, y in zip(x_train, y_train) if int(y) == 1]
    # Keep a compact explanation corpus for the demo (speed + repo size).
    spam_for_demo = spam_train_texts[:80]
    save_spam_corpus(spam_for_demo, MODELS_DIR / "spam_corpus.json")

    models = [] if local_only else [
        AlwaysSafeBaseline(),
        TfidfLogRegModel(),
        EmbeddingLogRegModel(),
    ]

    english_rows: list[dict] = []
    egypt_rows: list[dict] = []

    for model in models:
        print(f"\n=== Training {model.name} ===")
        model.fit(x_train, y_train)

        if model.name == "baseline":
            out = MODELS_DIR / "baseline.joblib"
        elif model.name == "model1_tfidf":
            out = MODELS_DIR / "model1_tfidf.joblib"
        else:
            out = MODELS_DIR / "model2_embed_lr.joblib"
        model.save(out)
        print(f"Saved {out.name}")

        eng_pred = model.predict(x_test)
        eng_metrics = spam_precision_recall(y_test, eng_pred.labels)
        eng_metrics["model"] = model.name
        eng_metrics["n_test"] = len(y_test)
        english_rows.append(eng_metrics)
        print(
            f"English spam precision={eng_metrics['precision_spam']:.3f} "
            f"recall={eng_metrics['recall_spam']:.3f}"
        )

        plot_confusion_matrix(
            y_test,
            eng_pred.labels,
            title=f"English test — {model.name}",
            out_path=RESULTS_DIR / f"english_confusion_{model.name}.png",
        )
        wrong_eng = wrong_predictions(
            x_test, y_test, eng_pred.labels, eng_pred.probabilities
        )
        write_csv(
            RESULTS_DIR / f"wrong_predictions_english_{model.name}.csv",
            wrong_eng,
            ["text", "true_label", "pred_label", "prob_spam"],
        )

        eg_pred = model.predict(eg_texts)
        breakdown = egypt_script_breakdown(eg_scripts, eg_labels, eg_pred.labels)
        for row in breakdown:
            row_out = {"model": model.name, **row}
            egypt_rows.append(row_out)
        overall = next(r for r in breakdown if r["script"] == "ALL")
        print(
            f"Egypt: scams caught {overall['scams_caught']}/{overall['scams_total']}, "
            f"false alarms {overall['false_alarms']}/{overall['safe_total']}"
        )
        plot_egypt_by_script(
            breakdown,
            RESULTS_DIR / f"egypt_by_script_{model.name}.png",
            model.name,
        )
        wrong_eg = wrong_predictions(
            eg_texts,
            eg_labels,
            eg_pred.labels,
            eg_pred.probabilities,
            scripts=eg_scripts,
        )
        write_csv(
            RESULTS_DIR / f"wrong_predictions_egypt_{model.name}.csv",
            wrong_eg,
            ["text", "true_label", "pred_label", "prob_spam", "script"],
        )
        conf = high_confidence_stats(eg_labels, eg_pred.labels, eg_pred.probabilities)
        print(
            f"Egypt high-confidence (>=80%): "
            f"{conf['n_correct']}/{conf['n_confident']} correct, "
            f"{conf['n_wrong']} wrong"
        )

    # Model 2 + local SMS. Test file is still never fitted.
    eg_extra_texts, eg_extra_labels, _eg_scripts = load_egypt_train_extra()
    world_texts, world_labels, _world_scripts = load_global_train_extra()
    extra_texts = eg_extra_texts + world_texts
    extra_labels = np.concatenate([eg_extra_labels, world_labels])
    overlap = set(extra_texts) & set(eg_texts)
    if overlap:
        raise RuntimeError(
            f"Train/test leakage: {len(overlap)} Egypt messages appear in both files."
        )
    repeated_texts: list[str] = []
    repeated_labels: list[int] = []
    for text, label in zip(extra_texts, extra_labels):
        times = LOCAL_REPEAT_SCAM if int(label) == 1 else LOCAL_REPEAT_SAFE
        repeated_texts.extend([text] * times)
        repeated_labels.extend([int(label)] * times)
    local_texts = x_train + repeated_texts
    local_labels = np.concatenate([y_train, np.asarray(repeated_labels, dtype=int)])
    print(
        f"\n=== Training model2_local "
        f"(English + {len(eg_extra_texts)} Egypt SMS + {len(world_texts)} global SMS, "
        f"scam x{LOCAL_REPEAT_SCAM}, safe x{LOCAL_REPEAT_SAFE}) ==="
    )
    local_model = EmbeddingLogRegModel()
    local_model.name = "model2_local"
    local_model.clf.epochs = 120
    local_model.fit(local_texts, local_labels)
    local_model.save(MODELS_DIR / "model2_local.joblib")
    print("Saved model2_local.joblib")

    eng_local = local_model.predict(x_test)
    eng_local_metrics = spam_precision_recall(y_test, eng_local.labels)
    eng_local_metrics["model"] = "model2_local"
    eng_local_metrics["n_test"] = len(y_test)
    english_rows.append(eng_local_metrics)
    print(
        f"English spam precision={eng_local_metrics['precision_spam']:.3f} "
        f"recall={eng_local_metrics['recall_spam']:.3f}"
    )
    plot_confusion_matrix(
        y_test,
        eng_local.labels,
        title="English test — model2_local",
        out_path=RESULTS_DIR / "english_confusion_model2_local.png",
    )

    eg_local = local_model.predict(eg_texts)
    local_breakdown = egypt_script_breakdown(eg_scripts, eg_labels, eg_local.labels)
    for row in local_breakdown:
        egypt_rows.append({"model": "model2_local", **row})
    overall = next(r for r in local_breakdown if r["script"] == "ALL")
    print(
        f"Egypt: scams caught {overall['scams_caught']}/{overall['scams_total']}, "
        f"false alarms {overall['false_alarms']}/{overall['safe_total']}"
    )
    conf = high_confidence_stats(eg_labels, eg_local.labels, eg_local.probabilities)
    print(
        f"Egypt high-confidence (>=80%): "
        f"{conf['n_correct']}/{conf['n_confident']} correct, "
        f"{conf['n_wrong']} wrong"
    )
    plot_egypt_by_script(
        local_breakdown,
        RESULTS_DIR / "egypt_by_script_model2_local.png",
        "model2_local",
    )
    wrong_local = wrong_predictions(
        eg_texts,
        eg_labels,
        eg_local.labels,
        eg_local.probabilities,
        scripts=eg_scripts,
    )
    write_csv(
        RESULTS_DIR / "wrong_predictions_egypt_model2_local.csv",
        wrong_local,
        ["text", "true_label", "pred_label", "prob_spam", "script"],
    )

    if local_only:
        def _keep(path: Path, rows: list[dict]) -> list[dict]:
            if not path.exists():
                return rows
            import csv

            with path.open(encoding="utf-8-sig", newline="") as f:
                old = [r for r in csv.DictReader(f) if r.get("model") != "model2_local"]
            # DictReader values are strings; new rows stay typed. Rewrite from old+new as strings is ok.
            return old + rows

        english_rows = _keep(RESULTS_DIR / "english_metrics.csv", english_rows)
        egypt_rows = _keep(RESULTS_DIR / "egypt_metrics.csv", egypt_rows)

    write_csv(
        RESULTS_DIR / "english_metrics.csv",
        english_rows,
        ["model", "n_test", "precision_spam", "recall_spam"],
    )
    write_csv(
        RESULTS_DIR / "egypt_metrics.csv",
        egypt_rows,
        [
            "model",
            "script",
            "scams_total",
            "scams_caught",
            "safe_total",
            "false_alarms",
        ],
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "english_train_size": len(x_train),
        "english_test_size": len(x_test),
        "egypt_test_size": len(eg_texts),
        "english_metrics": english_rows,
        "egypt_metrics": egypt_rows,
        "notes": [
            "egypt_test.csv was never used for training.",
            "model2_local also trains on data/egypt_train_extra.csv, repeated so it is not drowned out by English SMS.",
            "Baseline always predicts SAFE (not spam).",
            "Logistic regression and TF-IDF are implemented in NumPy/Torch "
            "(same idea as scikit-learn) so the project runs on machines where "
            "Windows Smart App Control blocks scipy/sklearn DLLs.",
            "Report numbers honestly; weak Arabic transfer is an expected finding.",
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDone. See results/ and models/.")
    for row in english_rows:
        print(
            f"  {row['model']}: P={float(row['precision_spam']):.3f} "
            f"R={float(row['recall_spam']):.3f}"
        )


if __name__ == "__main__":
    run(local_only="--local-only" in sys.argv)
