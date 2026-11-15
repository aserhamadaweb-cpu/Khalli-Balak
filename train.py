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
from pathlib import Path

from src.data_loading import (
    RANDOM_SEED,
    english_split,
    load_egypt_test,
    load_english_sms,
)
from src.evaluate import (
    egypt_script_breakdown,
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


def run() -> None:
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
    save_spam_corpus(spam_train_texts, MODELS_DIR / "spam_corpus.json")

    models = [
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
            "Egyptian messages were never used for training.",
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
            f"  {row['model']}: P={row['precision_spam']:.3f} R={row['recall_spam']:.3f}"
        )


if __name__ == "__main__":
    run()
