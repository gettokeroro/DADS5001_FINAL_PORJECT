"""
Phase 4 — Add Vitamin deficiency disease to scoring engine source data
=======================================================================

Usage (from project root):
    python scripts/add_vitamin_deficiency_to_train.py

Generic vitamin/mineral deficiency (D + B12 + Iron pattern) — chronic
fatigue + neuropsych + brittle nails are common across these.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "itachi_train.csv"

DISEASE_NAME = "Vitamin deficiency"
N_ROWS = 100

SYMPTOM_COUNTS: dict[str, int] = {
    "fatigue":              95,   # hallmark across vitamin/iron def
    "lethargy":             70,
    "weakness_in_limbs":    60,
    "brittle_nails":        60,   # iron def
    "mood_swings":          55,
    "depression":           50,
    "anxiety":              50,
    "dizziness":            45,
    "headache":             40,
    "cold_hands_and_feets": 40,   # anemia · poor circulation
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
        print(f"  {mark} {sym:22s}: {actual}/{len(target)} = {actual/len(target):.2f} (expected {expected_count/N_ROWS:.2f})")
        if not ok:
            all_ok = False
    if not all_ok:
        return 1
    print("\n✅ Done — push data/raw/itachi_train.csv to GitHub to deploy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
