"""Load English SMS spam data and the Egyptian test CSV (no pandas/datasets)."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from src.ml_utils import stratified_train_test_split

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EGYPT_TEST_PATH = PROJECT_ROOT / "data" / "egypt_test.csv"
ENGLISH_CACHE = PROJECT_ROOT / "data" / "sms_spam_english.csv"

# Official UCI SMS Spam Collection (same source as Hugging Face ucirvine/sms_spam).
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


def _parse_uci_bytes(raw: bytes) -> list[tuple[str, int]]:
    """Parse tab-separated 'ham/spam \\t message' lines."""
    text = raw.decode("utf-8", errors="replace")
    rows: list[tuple[str, int]] = []
    for line in text.splitlines():
        line = line.strip("\n\r")
        if not line:
            continue
        if "\t" not in line:
            continue
        label_str, msg = line.split("\t", 1)
        label_str = label_str.strip().lower()
        if label_str not in {"ham", "spam"}:
            continue
        rows.append((msg.strip(), 1 if label_str == "spam" else 0))
    return rows


def download_english_sms(force: bool = False) -> Path:
    """Download UCI SMS Spam Collection once and cache as CSV."""
    if ENGLISH_CACHE.exists() and not force:
        return ENGLISH_CACHE

    ENGLISH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    zip_path = ENGLISH_CACHE.with_suffix(".zip")
    print(f"Downloading English SMS dataset from UCI...\n  {UCI_ZIP_URL}")
    urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # File inside zip is usually 'SMSSpamCollection'
        names = zf.namelist()
        target = next(
            (n for n in names if n.rstrip("/").endswith("SMSSpamCollection")),
            names[0],
        )
        raw = zf.read(target)

    rows = _parse_uci_bytes(raw)
    if len(rows) < 1000:
        raise RuntimeError(f"Unexpected SMS count after parse: {len(rows)}")

    with ENGLISH_CACHE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    zip_path.unlink(missing_ok=True)
    print(f"Cached {len(rows)} English SMS -> {ENGLISH_CACHE}")
    return ENGLISH_CACHE


def load_english_sms() -> tuple[list[str], np.ndarray]:
    """Return English texts and labels (1=spam). Downloads on first run."""
    path = download_english_sms()
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.asarray(labels, dtype=int)


def load_egypt_test(
    path: Path | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load hand-written Egyptian test set. Never used for training."""
    csv_path = path or EGYPT_TEST_PATH
    texts: list[str] = []
    labels: list[int] = []
    scripts: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text", "label", "script"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError("egypt_test.csv must have columns: text,label,script")
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            scripts.append(row["script"].strip().lower())
    return texts, np.asarray(labels, dtype=int), scripts


def english_split(
    texts: list[str], labels: np.ndarray, seed: int = RANDOM_SEED
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    return stratified_train_test_split(texts, labels, test_size=0.2, seed=seed)
