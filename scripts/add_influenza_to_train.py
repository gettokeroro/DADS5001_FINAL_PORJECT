"""
Phase 4 — Add Influenza disease to scoring engine source data
==============================================================

Why this script exists:
-----------------------
`utils/scoring.py:load_artifacts()` builds the disease-symptom freq table
by groupby+mean over `data/raw/itachi_train.csv` (the Kaggle one-hot dataset),
NOT from `data/processed/disease_symptom_long.csv`. So adding rows to the
processed CSV has no effect on Top-3 results.

To add a new disease, we must append patient rows (one-hot encoded) to
itachi_train.csv. The freq DuckDB sees for each symptom of the new disease
will then be `n_positive / n_total_rows` for that disease.

Design (Influenza, 100 rows):
-----------------------------
| symptom      | n=1 | freq |
|--------------|-----|------|
| high_fever   |  95 | 0.95 |
| chills       |  90 | 0.90 |
| muscle_pain  |  95 | 0.95 |
| joint_pain   |  85 | 0.85 |
| headache     |  90 | 0.90 |
| fatigue      |  95 | 0.95 |
| malaise      |  90 | 0.90 |
| cough        |  85 | 0.85 |

All other 124 columns = 0. prognosis = "Influenza".

Idempotent: if Influenza already has rows, the script skips (no double-add).

Usage (from project root):
--------------------------
    python scripts/add_influenza_to_train.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "itachi_train.csv"

DISEASE_NAME = "Influenza"
N_ROWS = 100

# (symptom_column, n_rows_with_value_1)
SYMPTOM_COUNTS: dict[str, int] = {
    "high_fever":  95,
    "chills":      90,
    "muscle_pain": 95,
    "joint_pain":  85,
    "headache":    90,
    "fatigue":     95,
    "malaise":     90,
    "cough":       85,
}


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"❌ File not found: {TRAIN_CSV}")
        return 1

    print(f"Reading: {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    train = pd.read_csv(TRAIN_CSV)
    print(f"BEFORE: {len(train)} rows · "
          f"{train['prognosis'].nunique()} diseases")

    # Idempotency — don't double-add
    if (train["prognosis"] == DISEASE_NAME).any():
        n_existing = int((train["prognosis"] == DISEASE_NAME).sum())
        print(f"⚠ {DISEASE_NAME} already has {n_existing} rows · "
              f"skipping (script is idempotent)")
        return 0

    # Validate that all design symptoms are real columns
    symptom_cols = [c for c in train.columns if c != "prognosis"]
    missing = [s for s in SYMPTOM_COUNTS if s not in symptom_cols]
    if missing:
        print(f"❌ Symptoms not found in itachi_train.csv columns: {missing}")
        return 1

    # Build N_ROWS new rows · one-hot encoded · deterministic via seed
    rng = np.random.default_rng(seed=42)
    new_data: dict[str, list | np.ndarray] = {
        col: np.zeros(N_ROWS, dtype=int) for col in symptom_cols
    }
    for sym, n_positive in SYMPTOM_COUNTS.items():
        if n_positive > N_ROWS:
            print(f"❌ {sym}: n_positive ({n_positive}) > N_ROWS ({N_ROWS})")
            return 1
        # Pick n_positive distinct row indices (deterministic with seed)
        idx = rng.permutation(N_ROWS)[:n_positive]
        new_data[sym][idx] = 1
    new_data["prognosis"] = [DISEASE_NAME] * N_ROWS

    # Preserve column order from original
    new_rows = pd.DataFrame(new_data, columns=train.columns)

    # Append + write back
    out = pd.concat([train, new_rows], ignore_index=True)
    out.to_csv(TRAIN_CSV, index=False)
    print(f"AFTER:  {len(out)} rows · "
          f"{out['prognosis'].nunique()} diseases")
    print(f"Wrote:  {TRAIN_CSV.relative_to(PROJECT_ROOT)}")

    # Verify freq pattern came out exactly as designed
    flu_only = out[out["prognosis"] == DISEASE_NAME]
    print(f"\nVerify freq for {DISEASE_NAME} ({len(flu_only)} rows):")
    all_ok = True
    for sym, expected_count in SYMPTOM_COUNTS.items():
        actual_count = int(flu_only[sym].sum())
        actual_freq = actual_count / len(flu_only)
        expected_freq = expected_count / N_ROWS
        ok = actual_count == expected_count
        mark = "✓" if ok else "✗"
        print(f"  {mark} {sym:13s}: "
              f"{actual_count}/{len(flu_only)} = {actual_freq:.2f} "
              f"(expected {expected_freq:.2f})")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n❌ Some freq values don't match design — check seed / logic")
        return 1

    print("\n✅ Done — push data/raw/itachi_train.csv to GitHub to deploy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
