"""
Phase 4 — Add Acute Bronchitis disease to scoring engine source data
=====================================================================

Usage (from project root):
    python scripts/add_acute_bronchitis_to_train.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "itachi_train.csv"

DISEASE_NAME = "Acute bronchitis"
N_ROWS = 100

SYMPTOM_COUNTS: dict[str, int] = {
    "cough":           100,   # defining symptom · usually persistent
    "phlegm":           85,   # productive cough common
    "malaise":          65,
    "mild_fever":       60,
    "fatigue":          55,
    "chest_pain":       50,   # pleuritic-like discomfort with cough
    "breathlessness":   40,   # mild dyspnea
    "headache":         35,
}


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"❌ File not found: {TRAIN_CSV}")
        return 1
    print(f"Reading: {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    train = pd.read_csv(TRAIN_CSV)
    print(f"BEFORE: {len(train)} rows · {train['prognosis'].nunique()} diseases")
    if (train["prognosis"] == DISEASE_NAME).any():
        n_existing = int((train["prognosis"] == DISEASE_NAME).sum())
        print(f"⚠ {DISEASE_NAME} already has {n_existing} rows · skipping")
        return 0
    symptom_cols = [c for c in train.columns if c != "prognosis"]
    missing = [s for s in SYMPTOM_COUNTS if s not in symptom_cols]
    if missing:
        print(f"❌ Symptoms not in columns: {missing}")
        return 1
    rng = np.random.default_rng(seed=42)
    new_data: dict[str, list | np.ndarray] = {
        col: np.zeros(N_ROWS, dtype=int) for col in symptom_cols
    }
    for sym, n_positive in SYMPTOM_COUNTS.items():
        if n_positive > N_ROWS:
            print(f"❌ {sym}: n_positive > N_ROWS")
            return 1
        idx = rng.permutation(N_ROWS)[:n_positive]
        new_data[sym][idx] = 1
    new_data["prognosis"] = [DISEASE_NAME] * N_ROWS
    new_rows = pd.DataFrame(new_data, columns=train.columns)
    out = pd.concat([train, new_rows], ignore_index=True)
    out.to_csv(TRAIN_CSV, index=False)
    print(f"AFTER:  {len(out)} rows · {out['prognosis'].nunique()} diseases")
    target = out[out["prognosis"] == DISEASE_NAME]
    print(f"\nVerify freq for {DISEASE_NAME} ({len(target)} rows):")
    all_ok = True
    for sym, expected_count in SYMPTOM_COUNTS.items():
        actual = int(target[sym].sum())
        ok = actual == expected_count
        mark = "✓" if ok else "✗"
        print(f"  {mark} {sym:18s}: {actual}/{len(target)} = {actual/len(target):.2f} (expected {expected_count/N_ROWS:.2f})")
        if not ok:
            all_ok = False
    if not all_ok:
        return 1
    print("\n✅ Done — push data/raw/itachi_train.csv to GitHub to deploy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
