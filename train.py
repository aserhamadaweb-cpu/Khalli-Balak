"""
Train and evaluate Khalli Balak scam-SMS classifiers.

Models:
  1) Baseline — always predict SAFE
  2) Model 1 — TF-IDF + Logistic Regression (English only)
  3) Model 2 — multilingual MiniLM embeddings + LR (English only)
  4) Model 2 local — embeddings + LR on English + Egypt/global extras
  5) Model 3 intent — embeddings + intent features on English + hard negatives

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
    load_hard_negatives,
    load_english_sms,
)
from src.evaluate import (
    classification_report_row,
    egypt_script_breakdown,
    high_confidence_stats,
    plot_confusion_matrix,
    plot_egypt_by_script,
    safe_money_false_positives,
    spam_precision_recall,
    write_csv,
    wrong_predictions,
)
from src.explain import save_spam_corpus
from src.models import (
    AlwaysSafeBaseline,
    EmbeddingLogRegModel,
    IntentEmbeddingLogRegModel,
    TfidfLogRegModel,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
# Repeat local SMS so 4,460 English rows do not drown them out.
LOCAL_REPEAT_SCAM = 12
LOCAL_REPEAT_SAFE = 40
# Hard negatives are the point of Model 3 — repeat so money-notice intent sticks.
HARD_REPEAT_SCAM = 20
HARD_REPEAT_SAFE = 35


def _assert_no_leakage(train_texts: list[str], test_texts: list[str], label: str) -> None:
    overlap = set(train_texts) & set(test_texts)
    if overlap:
        raise RuntimeError(
            f"Train/test leakage in {label}: {len(overlap)} messages appear in both files."
        )


def _eval_egypt(
    model_name: str,
    model,
    eg_texts: list[str],
    eg_labels: np.ndarray,
    eg_scripts: list[str],
    egypt_rows: list[dict],
    compare_rows: list[dict],
    money_fp_rows: list[dict],
) -> None:
    eg_pred = model.predict(eg_texts)
    breakdown = egypt_script_breakdown(eg_scripts, eg_labels, eg_pred.labels)
    for row in breakdown:
        egypt_rows.append({"model": model_name, **row})
    overall = next(r for r in breakdown if r["script"] == "ALL")
    print(
        f"Egypt: scams caught {overall['scams_caught']}/{overall['scams_total']}, "
        f"false alarms {overall['false_alarms']}/{overall['safe_total']}"
    )
    plot_egypt_by_script(
        breakdown,
        RESULTS_DIR / f"egypt_by_script_{model_name}.png",
        model_name,
    )
    plot_confusion_matrix(
        eg_labels,
        eg_pred.labels,
        title=f"Egypt test — {model_name}",
        out_path=RESULTS_DIR / f"egypt_confusion_{model_name}.png",
    )
    report = classification_report_row(model_name, eg_labels, eg_pred.labels)
    compare_rows.append(report)
    print(
        f"Egypt metrics: acc={report['accuracy']:.3f} "
        f"P={report['precision_spam']:.3f} "
        f"R={report['recall_spam']:.3f} "
        f"F1={report['f1_spam']:.3f}"
    )
    wrong_eg = wrong_predictions(
        eg_texts,
        eg_labels,
        eg_pred.labels,
        eg_pred.probabilities,
        scripts=eg_scripts,
    )
    write_csv(
        RESULTS_DIR / f"wrong_predictions_egypt_{model_name}.csv",
        wrong_eg,
        ["text", "true_label", "pred_label", "prob_spam", "script"],
    )
    money_fps = safe_money_false_positives(
        eg_texts,
        eg_labels,
        eg_pred.labels,
        eg_pred.probabilities,
        scripts=eg_scripts,
    )
    for row in money_fps:
        money_fp_rows.append({"model": model_name, **row})
    print(f"SAFE money false positives: {len(money_fps)}")
    conf = high_confidence_stats(
        eg_labels, eg_pred.labels, eg_pred.probabilities, min_confidence=0.75
    )
    print(
        f"Egypt high-confidence (>=75%): "
        f"{conf['n_correct']}/{conf['n_confident']} correct, "
        f"{conf['n_wrong']} wrong"
    )


def run(local_only: bool = False, intent_only: bool = False) -> None:
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
    spam_for_demo = spam_train_texts[:80]
    save_spam_corpus(spam_for_demo, MODELS_DIR / "spam_corpus.json")

    english_rows: list[dict] = []
    egypt_rows: list[dict] = []
    compare_rows: list[dict] = []
    money_fp_rows: list[dict] = []

    if not local_only and not intent_only:
        models = [
            AlwaysSafeBaseline(),
            TfidfLogRegModel(),
            EmbeddingLogRegModel(),
        ]
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
            english_rows.append(
                {
                    "model": model.name,
                    "n_test": len(y_test),
                    "precision_spam": eng_metrics["precision_spam"],
                    "recall_spam": eng_metrics["recall_spam"],
                }
            )
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
            _eval_egypt(
                model.name,
                model,
                eg_texts,
                eg_labels,
                eg_scripts,
                egypt_rows,
                compare_rows,
                money_fp_rows,
            )

    if not intent_only:
        # Model 2 + local SMS. Test file is still never fitted.
        eg_extra_texts, eg_extra_labels, _eg_scripts = load_egypt_train_extra()
        world_texts, world_labels, _world_scripts = load_global_train_extra()
        extra_texts = eg_extra_texts + world_texts
        extra_labels = np.concatenate([eg_extra_labels, world_labels])
        _assert_no_leakage(extra_texts, eg_texts, "egypt/global extras")
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
        english_rows.append(
            {
                "model": "model2_local",
                "n_test": len(y_test),
                "precision_spam": eng_local_metrics["precision_spam"],
                "recall_spam": eng_local_metrics["recall_spam"],
            }
        )
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
        _eval_egypt(
            "model2_local",
            local_model,
            eg_texts,
            eg_labels,
            eg_scripts,
            egypt_rows,
            compare_rows,
            money_fp_rows,
        )

    # Model 3: embeddings + intent on English + hard negatives (never egypt_test).
    hard_texts, hard_labels, _hard_scripts = load_hard_negatives()
    _assert_no_leakage(hard_texts, eg_texts, "hard_negatives_train")
    hard_rep_texts: list[str] = []
    hard_rep_labels: list[int] = []
    for text, label in zip(hard_texts, hard_labels):
        times = HARD_REPEAT_SCAM if int(label) == 1 else HARD_REPEAT_SAFE
        hard_rep_texts.extend([text] * times)
        hard_rep_labels.extend([int(label)] * times)
    intent_texts = x_train + hard_rep_texts
    intent_labels = np.concatenate([y_train, np.asarray(hard_rep_labels, dtype=int)])
    n_safe_hard = int((hard_labels == 0).sum())
    n_scam_hard = int((hard_labels == 1).sum())
    print(
        f"\n=== Training model3_intent "
        f"(English + {n_safe_hard} SAFE / {n_scam_hard} SCAM hard negatives, "
        f"scam x{HARD_REPEAT_SCAM}, safe x{HARD_REPEAT_SAFE}) ==="
    )
    intent_model = IntentEmbeddingLogRegModel()
    intent_model.fit(intent_texts, intent_labels)
    intent_model.save(MODELS_DIR / "model3_intent.joblib")
    print("Saved model3_intent.joblib")

    eng_intent = intent_model.predict(x_test)
    eng_intent_metrics = spam_precision_recall(y_test, eng_intent.labels)
    english_rows.append(
        {
            "model": "model3_intent",
            "n_test": len(y_test),
            "precision_spam": eng_intent_metrics["precision_spam"],
            "recall_spam": eng_intent_metrics["recall_spam"],
        }
    )
    print(
        f"English spam precision={eng_intent_metrics['precision_spam']:.3f} "
        f"recall={eng_intent_metrics['recall_spam']:.3f}"
    )
    plot_confusion_matrix(
        y_test,
        eng_intent.labels,
        title="English test — model3_intent",
        out_path=RESULTS_DIR / "english_confusion_model3_intent.png",
    )
    _eval_egypt(
        "model3_intent",
        intent_model,
        eg_texts,
        eg_labels,
        eg_scripts,
        egypt_rows,
        compare_rows,
        money_fp_rows,
    )

    # When only training the new model, re-score saved old models for a fair table.
    if intent_only:
        for name, path, loader in [
            ("baseline", MODELS_DIR / "baseline.joblib", AlwaysSafeBaseline.load),
            ("model1_tfidf", MODELS_DIR / "model1_tfidf.joblib", TfidfLogRegModel.load),
            (
                "model2_embed",
                MODELS_DIR / "model2_embed_lr.joblib",
                EmbeddingLogRegModel.load,
            ),
            (
                "model2_local",
                MODELS_DIR / "model2_local.joblib",
                EmbeddingLogRegModel.load,
            ),
        ]:
            if not path.exists():
                print(f"Skip compare for missing {path.name}")
                continue
            print(f"\n=== Re-scoring saved {name} on Egypt test ===")
            old = loader(path)
            if hasattr(old, "name"):
                old.name = name
            _eval_egypt(
                name,
                old,
                eg_texts,
                eg_labels,
                eg_scripts,
                egypt_rows,
                compare_rows,
                money_fp_rows,
            )

    if local_only or intent_only:
        def _keep(path: Path, rows: list[dict], drop_models: set[str]) -> list[dict]:
            if not path.exists():
                return rows
            import csv

            with path.open(encoding="utf-8-sig", newline="") as f:
                old = [r for r in csv.DictReader(f) if r.get("model") not in drop_models]
            return old + rows

        if intent_only:
            english_rows = _keep(
                RESULTS_DIR / "english_metrics.csv",
                english_rows,
                {"model3_intent"},
            )
        elif local_only:
            english_rows = _keep(
                RESULTS_DIR / "english_metrics.csv",
                english_rows,
                {"model2_local"},
            )
            egypt_rows = _keep(
                RESULTS_DIR / "egypt_metrics.csv",
                egypt_rows,
                {"model2_local"},
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
    write_csv(
        RESULTS_DIR / "egypt_compare_old_vs_new.csv",
        compare_rows,
        [
            "model",
            "n_test",
            "accuracy",
            "precision_spam",
            "recall_spam",
            "f1_spam",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    write_csv(
        RESULTS_DIR / "egypt_safe_money_false_positives.csv",
        money_fp_rows,
        ["model", "text", "true_label", "pred_label", "prob_spam", "script"],
    )
    money_fp_summary = []
    for model_name in sorted({r["model"] for r in money_fp_rows} | {r["model"] for r in compare_rows}):
        count = sum(1 for r in money_fp_rows if r["model"] == model_name)
        money_fp_summary.append({"model": model_name, "safe_money_false_positives": count})
    # Include zeros for models with no money FPs.
    for row in compare_rows:
        if row["model"] not in {r["model"] for r in money_fp_summary}:
            money_fp_summary.append(
                {"model": row["model"], "safe_money_false_positives": 0}
            )
    # Deduplicate summary
    seen = set()
    deduped = []
    for row in money_fp_summary:
        if row["model"] in seen:
            continue
        seen.add(row["model"])
        deduped.append(row)
    write_csv(
        RESULTS_DIR / "egypt_safe_money_fp_summary.csv",
        deduped,
        ["model", "safe_money_false_positives"],
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "english_train_size": len(x_train),
        "english_test_size": len(x_test),
        "egypt_test_size": len(eg_texts),
        "hard_negatives_safe": n_safe_hard,
        "hard_negatives_scam": n_scam_hard,
        "english_metrics": english_rows,
        "egypt_metrics": egypt_rows,
        "egypt_compare_old_vs_new": compare_rows,
        "safe_money_false_positives_summary": deduped,
        "notes": [
            "egypt_test.csv was never used for training.",
            "model3_intent trains on UCI English SMS + data/hard_negatives_train.csv only.",
            "Hard-negative texts are self-written and checked for no overlap with egypt_test.csv.",
            "Intent features: share-code, do-not-share, link/unofficial link, urgency, prize, send-money, pure notice.",
            "Baseline always predicts SAFE (not spam).",
            "Report numbers honestly; remaining mistakes stay listed in wrong_predictions_*.csv.",
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDone. See results/ and models/.")
    for row in compare_rows:
        print(
            f"  {row['model']}: acc={float(row['accuracy']):.3f} "
            f"P={float(row['precision_spam']):.3f} "
            f"R={float(row['recall_spam']):.3f} "
            f"F1={float(row['f1_spam']):.3f} "
            f"FP={int(row['fp'])}"
        )


if __name__ == "__main__":
    args = set(sys.argv[1:])
    run(local_only="--local-only" in args, intent_only="--intent-only" in args)
