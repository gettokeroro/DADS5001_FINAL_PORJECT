"""
scripts/patch_symptom_alias_mongo.py
=====================================
One-off patch: sync corrected `symptom_th_alt` for `headache` and `dizziness`
from the local CSV up to the MongoDB Atlas `symptom_dictionary` collection.

WHY: pages load the dictionary cloud-first (MongoDB → CSV fallback). Editing the
CSV alone does NOT change the live app — the deployed app reads from Mongo.
Run this once locally (network can reach Atlas) after editing the CSV.

Fix applied:
  - headache : removed bare alias "ความดัน" (kept ความดันสูง/โลหิตสูง/ขึ้น)
  - dizziness: added "ความดันต่ำ, ความดันตก"
So typing "ความดันต่ำ" → เวียนหัว (dizziness) instead of ปวดหัว (headache).

USAGE (Windows, Anaconda):
    cd "<project root>"
    python scripts/patch_symptom_alias_mongo.py

Reads Mongo URI from .streamlit/secrets.toml  ([mongodb] uri, db_name).
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data" / "processed" / "symptom_dictionary_th.csv"
SECRETS = ROOT / ".streamlit" / "secrets.toml"
TARGETS = ["headache", "dizziness"]


def _load_secrets() -> dict:
    try:
        import tomllib  # Python 3.11+
        with open(SECRETS, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        import toml  # pip install toml
        return toml.load(str(SECRETS))


def main() -> int:
    if not CSV.exists():
        print(f"[ERR] CSV not found: {CSV}")
        return 1
    if not SECRETS.exists():
        print(f"[ERR] secrets not found: {SECRETS}")
        return 1

    df = pd.read_csv(CSV, encoding="utf-8-sig")
    new_alt = {}
    for en in TARGETS:
        row = df[df["symptom_en"] == en]
        if row.empty:
            print(f"[ERR] '{en}' not in CSV — aborting")
            return 1
        new_alt[en] = str(row.iloc[0].get("symptom_th_alt") or "")
        print(f"[CSV] {en}.symptom_th_alt = {new_alt[en]!r}")

    creds = _load_secrets().get("mongodb", {})
    uri = (creds.get("uri") or "").strip()
    db_name = creds.get("db_name", "dads5001")
    if not uri:
        print("[ERR] [mongodb].uri missing in secrets.toml")
        return 1

    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    col = client[db_name]["symptom_dictionary"]
    print(f"[OK] connected · db={db_name} · symptom_dictionary count="
          f"{col.count_documents({})}")

    for en in TARGETS:
        before = col.find_one({"symptom_en": en}, {"_id": 0, "symptom_th_alt": 1})
        print(f"\n[{en}] BEFORE: {before.get('symptom_th_alt') if before else 'MISSING'}")
        res = col.update_one(
            {"symptom_en": en},
            {"$set": {"symptom_th_alt": new_alt[en]}},
        )
        after = col.find_one({"symptom_en": en}, {"_id": 0, "symptom_th_alt": 1})
        print(f"[{en}] matched={res.matched_count} modified={res.modified_count}")
        print(f"[{en}] AFTER : {after.get('symptom_th_alt') if after else 'MISSING'}")

    print("\n[DONE] MongoDB updated. Reboot the Streamlit app (clears cache) to see the change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
