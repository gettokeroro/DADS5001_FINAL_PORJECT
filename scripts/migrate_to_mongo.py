#!/usr/bin/env python3
"""
Phase 9b — One-time migration: upload 2 collections to MongoDB Atlas

Collections:
    symptom_dictionary  (132 documents · from data/processed/symptom_dictionary_th.csv)
    disease_prevalence  (48 documents  · from data/processed/disease_prevalence.csv)

Usage:
    # Step 1 — set credentials (pick one way):
    #   a) env var:
    #      Windows: set MONGO_URI=mongodb+srv://user:pass@cluster0.xkwi2nj.mongodb.net/?appName=Cluster0
    #      Mac/Linux: export MONGO_URI="mongodb+srv://..."
    #   b) fill in .streamlit/secrets.toml with [mongodb].uri
    #
    # Step 2 — run:
    python scripts/migrate_to_mongo.py              # upload + verify
    python scripts/migrate_to_mongo.py --verify     # verify only (no upload)
    python scripts/migrate_to_mongo.py --dry-run    # show row counts only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_NAME = "dads5001"

CSV_MAP: dict[str, Path] = {
    "symptom_dictionary": ROOT / "data" / "processed" / "symptom_dictionary_th.csv",
    "disease_prevalence": ROOT / "data" / "processed" / "disease_prevalence.csv",
}

EXPECTED_COUNTS: dict[str, int] = {
    "symptom_dictionary": 132,
    "disease_prevalence": 48,
}


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------
def _get_uri() -> str:
    """Priority: MONGO_URI env → .streamlit/secrets.toml [mongodb].uri"""
    uri = os.environ.get("MONGO_URI", "").strip()
    if uri:
        return uri

    toml_path = ROOT / ".streamlit" / "secrets.toml"
    if toml_path.exists():
        try:
            try:
                import tomllib  # Python 3.11+ stdlib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(toml_path, "rb") as f:
                secrets = tomllib.load(f)
            uri = secrets.get("mongodb", {}).get("uri", "").strip()
            if uri:
                return uri
        except Exception as e:
            print(f"⚠ Could not read secrets.toml: {e}")

    raise ValueError(
        "\n❌  MongoDB URI not found.\n"
        "Option A — set env var:\n"
        '  Windows:    set MONGO_URI="mongodb+srv://user:pass@cluster0.xkwi2nj.mongodb.net/"\n'
        '  Mac/Linux:  export MONGO_URI="mongodb+srv://..."\n'
        "Option B — fill in .streamlit/secrets.toml:\n"
        "  [mongodb]\n"
        '  uri = "mongodb+srv://user:pass@cluster0.xkwi2nj.mongodb.net/"\n'
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _upload_collection(db, name: str, csv_path: Path, drop_first: bool = True) -> int:
    """Read CSV → upload to collection. Returns number of inserted docs."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Replace NaN with None so pymongo serialises correctly
    records = df.where(pd.notnull(df), None).to_dict("records")

    col = db[name]
    if drop_first:
        col.drop()
        print(f"    Dropped existing '{name}' collection")

    result = col.insert_many(records)
    return len(result.inserted_ids)


def _verify_collection(db, name: str, expected: int) -> bool:
    """Count documents and compare to expected. Returns True if ok."""
    actual = db[name].count_documents({})
    ok = actual >= expected
    icon = "✅" if ok else "❌"
    print(f"    {icon} {name}: {actual} docs (expected ≥{expected})")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate DADS5001 CSVs → MongoDB Atlas (Phase 9b)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check document counts only — no upload"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print CSV shapes only — no upload, no connection"
    )
    args = parser.parse_args()

    # ── Dry-run: no network needed ──────────────────────────────────────────
    if args.dry_run:
        print("── Dry-run (no connection) ──")
        for col_name, csv_path in CSV_MAP.items():
            if not csv_path.exists():
                print(f"  ❌ Not found: {csv_path}")
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            print(f"  {col_name}: {len(df)} rows · columns: {list(df.columns)}")
        return

    # ── Connect ─────────────────────────────────────────────────────────────
    try:
        from pymongo import MongoClient
    except ImportError:
        print("❌ pymongo not installed.\nRun: pip install 'pymongo>=4.6,<5.0'")
        sys.exit(1)

    uri = _get_uri()
    print("Connecting to MongoDB Atlas...")
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        client.admin.command("ping")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    print("✅ Connected\n")

    db = client[DB_NAME]

    # ── Verify only ─────────────────────────────────────────────────────────
    if args.verify:
        print("── Verify ──")
        all_ok = all(
            _verify_collection(db, name, exp)
            for name, exp in EXPECTED_COUNTS.items()
        )
        print()
        if all_ok:
            print("✅ All collections verified — MongoDB ready for Phase 9b!")
        else:
            print("❌ Some collections have fewer docs than expected.")
            print("   Run without --verify to re-upload.")
        client.close()
        return

    # ── Upload ───────────────────────────────────────────────────────────────
    print("── Uploading collections ──")
    any_error = False
    for col_name, csv_path in CSV_MAP.items():
        print(f"\n[{col_name}]")
        if not csv_path.exists():
            print(f"  ❌ CSV not found: {csv_path}")
            any_error = True
            continue
        try:
            n = _upload_collection(db, col_name, csv_path, drop_first=True)
            print(f"  ✅ Inserted {n} documents")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            any_error = True

    print("\n── Verify after upload ──")
    all_ok = all(
        _verify_collection(db, name, exp)
        for name, exp in EXPECTED_COUNTS.items()
    )

    print()
    if all_ok and not any_error:
        print("🎉 Migration complete! Collections are live on MongoDB Atlas.")
        print("   Next: set [mongodb].uri in Streamlit Cloud → Settings → Secrets")
    else:
        print("⚠ Migration finished with some issues — check output above.")

    client.close()


if __name__ == "__main__":
    main()
