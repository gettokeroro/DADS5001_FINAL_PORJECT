"""
Phase 4 — Add Pharyngitis/Tonsillitis disease to scoring engine source data
============================================================================

Usage (from project root):
    python scripts/add_pharyngitis_to_train.py

Idempotent · seed=42 deterministic. See add_influenza_to_train.py for the
underlying rationale (itachi_train.csv is the real source — not the
processed CSV).
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "itachi_train.csv"

DISEASE_NAME = "Pharyngitis"
N_ROWS = 100

# (symptom_column, n_rows_with_value_1)
SYMPTOM_COUNTS: dict[str, int] = {
    "throat_irritation":   100,   # defining symptom — sore throat
    "swelled_lymph_nodes":  75,   # anterior cervical · strep marker
    "patches_in_throat":    70,   # strep marker · viral 30% no patches
    "malaise":              70,
    "headache":             65,
    "mild_fever":           60,   # most common · low-grade
    "fatigue":              55,
    "high_fever":           30,   # bacterial strep more common
    "cough":                25,   # viral pharyngitis only · low for bacterial
}


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"❌ File not found: {TRAIN_CSV}")
        return 1

    print(f"Reading: {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    train = pd.read_csv(TRAIN_CSV)
    print(f"BEFORE: {len(train)} rows · "
          f"{train['prognosis'].nunique()} diseases")

    if (train["prognosis"] == DISEASE_NAME).any():
        n_existing = int((train["prognosis"] == DISEASE_NAME).sum())
        print(f"⚠ {DISEASE_NAME} already has {n_existing} rows · "
              f"skipping (script is idempotent)")
        return 0

    symptom_cols = [c for c in train.columns if c != "prognosis"]
    missing = [s for s in SYMPTOM_COUNTS if s not in symptom_cols]
    if missing:
        print(f"❌ Symptoms not found in itachi_train.csv columns: {missing}")
        return 1

    rng = np.random.default_rng(seed=42)
    new_data: dict[str, list | np.ndarray] = {
        col: np.zeros(N_ROWS, dtype=int) for col in symptom_cols
    }
    for sym, n_positive in SYMPTOM_COUNTS.items():
        if n_positive > N_ROWS:
            print(f"❌ {sym}: n_positive ({n_positive}) > N_ROWS ({N_ROWS})")
            return 1
        idx = rng.permutation(N_ROWS)[:n_positive]
        new_data[sym][idx] = 1
    new_data["prognosis"] = [DISEASE_NAME] * N_ROWS

    new_rows = pd.DataFrame(new_data, columns=train.columns)
    out = pd.concat([train, new_rows], ignore_index=True)
    out.to_csv(TRAIN_CSV, index=False)
    print(f"AFTER:  {len(out)} rows · "
          f"{out['prognosis'].nunique()} diseases")
    print(f"Wrote:  {TRAIN_CSV.relative_to(PROJECT_ROOT)}")

    target = out[out["prognosis"] == DISEASE_NAME]
    print(f"\nVerify freq for {DISEASE_NAME} ({len(target)} rows):")
    all_ok = True
    for sym, expected_count in SYMPTOM_COUNTS.items():
        actual_count = int(target[sym].sum())
        actual_freq = actual_count / len(target)
        expected_freq = expected_count / N_ROWS
        ok = actual_count == expected_count
        mark = "✓" if ok else "✗"
        print(f"  {mark} {sym:21s}: "
              f"{actual_count}/{len(target)} = {actual_freq:.2f} "
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
