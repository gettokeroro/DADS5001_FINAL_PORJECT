"""
Snowflake migration / verification CLI for DADS5001 Final Project.

Reads credentials from .streamlit/secrets.toml [snowflake] section (gitignored).
Provides two modes:
    --verify  : read-only row+column count checks across all 3 tables
    --reload  : TRUNCATE + re-upload CSV via PUT/COPY INTO (requires --confirm)

This script is idempotent and safe to re-run. Verify mode performs no writes.

Usage examples:
    python scripts/migrate_to_snowflake.py --verify
    python scripts/migrate_to_snowflake.py --reload --confirm
    python scripts/migrate_to_snowflake.py --reload --table TRAIN_MATRIX --confirm

Prerequisites:
    pip install snowflake-connector-python tomli
    Edit .streamlit/secrets.toml and fill in [snowflake] section
    (see .streamlit/secrets.toml.example for template)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# tomllib is stdlib in Python 3.11+; tomli is the backport for 3.10-
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        print("ERROR: TOML parser missing.")
        print("       Run: pip install tomli   (only needed on Python < 3.11)")
        sys.exit(1)

try:
    import snowflake.connector
except ImportError:
    print("ERROR: snowflake-connector-python not installed.")
    print("       Run: pip install snowflake-connector-python")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

# (csv_rel_path, snowflake_table, expected_rows, expected_cols)
TABLES: list[tuple[str, str, int, int]] = [
    ("data/raw/itachi_train.csv",                      "TRAIN_MATRIX", 5620, 133),
    ("data/processed/disease_drug_mapping_v2_ed.csv",  "DISEASE_DRUG",  109,  11),
    ("data/processed/hospitals_thailand.csv",          "HOSPITALS",    1581,   9),
]

REQUIRED_KEYS = {"account", "user", "password", "warehouse", "database", "schema"}


# ----------------------------------------------------------------------
# secrets + connection
# ----------------------------------------------------------------------

def load_secrets() -> dict:
    """Read [snowflake] block from .streamlit/secrets.toml."""
    if not SECRETS_PATH.exists():
        print(f"ERROR: {SECRETS_PATH} not found.")
        print("       Copy .streamlit/secrets.toml.example -> .streamlit/secrets.toml")
        print("       and fill in the [snowflake] section.")
        sys.exit(1)

    with SECRETS_PATH.open("rb") as f:
        data = tomllib.load(f)

    if "snowflake" not in data:
        print(f"ERROR: [snowflake] section missing in {SECRETS_PATH}")
        print("       See .streamlit/secrets.toml.example for required keys.")
        sys.exit(1)

    creds = data["snowflake"]
    missing = REQUIRED_KEYS - set(creds.keys())
    if missing:
        print(f"ERROR: missing keys in [snowflake] block: {sorted(missing)}")
        sys.exit(1)

    return creds


def connect(creds: dict):
    """Open Snowflake connection."""
    return snowflake.connector.connect(
        account=creds["account"],
        user=creds["user"],
        password=creds["password"],
        warehouse=creds["warehouse"],
        database=creds["database"],
        schema=creds["schema"],
    )


# ----------------------------------------------------------------------
# verify mode
# ----------------------------------------------------------------------

def verify(conn) -> bool:
    """Check row + column counts for all 3 tables. Returns True if all pass."""
    print("=" * 64)
    print("VERIFY MODE - row + column counts (read-only)")
    print("=" * 64)

    all_ok = True
    cur = conn.cursor()
    try:
        for _csv_path, table, exp_rows, exp_cols in TABLES:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                actual_rows = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = CURRENT_SCHEMA() AND TABLE_NAME = %s",
                    (table,),
                )
                actual_cols = cur.fetchone()[0]

                rows_ok = actual_rows == exp_rows
                cols_ok = actual_cols == exp_cols
                mark = "OK  " if (rows_ok and cols_ok) else "FAIL"
                print(
                    f"  [{mark}] {table:14s}  "
                    f"rows={actual_rows:>5} (exp {exp_rows:>5})  "
                    f"cols={actual_cols:>4} (exp {exp_cols:>4})"
                )
                if not (rows_ok and cols_ok):
                    all_ok = False
            except Exception as e:
                print(f"  [FAIL] {table}: {e}")
                all_ok = False
    finally:
        cur.close()

    print("=" * 64)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    print("=" * 64)
    return all_ok


# ----------------------------------------------------------------------
# reload mode (TRUNCATE + COPY INTO)
# ----------------------------------------------------------------------

def reload_table(conn, table: str, csv_path: Path, expected_rows: int) -> bool:
    """TRUNCATE the table, PUT local CSV to user stage, then COPY INTO."""
    print(f"\n--- Reloading {table} <-- {csv_path.name} ---")
    cur = conn.cursor()
    try:
        # 1. TRUNCATE existing data
        cur.execute(f"TRUNCATE TABLE {table}")
        print(f"    [1/3] TRUNCATE TABLE {table} ... done")

        # 2. PUT CSV to user stage. as_posix() ensures forward slashes for Snowflake.
        stage_path = f"@~/{table.lower()}_staging"

        # Remove any stale staged files (ignore failure on first run)
        try:
            cur.execute(f"REMOVE {stage_path}")
        except Exception:
            pass

        put_uri = f"file://{csv_path.resolve().as_posix()}"
        cur.execute(
            f"PUT {put_uri} {stage_path} "
            f"AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
        )
        print(f"    [2/3] PUT to {stage_path} ... done")

        # 3. COPY INTO with inline file format
        # EMPTY_FIELD_AS_NULL handles HOSPITALS.beds blank cell.
        # FIELD_OPTIONALLY_ENCLOSED_BY '"' handles Thai text + commas inside fields.
        cur.execute(
            f"COPY INTO {table} "
            f"FROM {stage_path} "
            "FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 "
            'FIELD_OPTIONALLY_ENCLOSED_BY = \'"\' '
            "EMPTY_FIELD_AS_NULL = TRUE "
            "ENCODING = 'UTF8' "
            "REPLACE_INVALID_CHARACTERS = FALSE) "
            "ON_ERROR = ABORT_STATEMENT"
        )
        copy_result = cur.fetchall()
        print(f"    [3/3] COPY INTO ... result: {copy_result}")

        # Clean up staged files
        try:
            cur.execute(f"REMOVE {stage_path}")
        except Exception:
            pass

        # Verify row count
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cur.fetchone()[0]
        if actual == expected_rows:
            print(f"    [OK ] {table} now has {actual} rows (matches expected)")
            return True
        print(f"    [FAIL] {table} has {actual} rows, expected {expected_rows}")
        return False
    except Exception as e:
        print(f"    [FAIL] {table}: {e}")
        return False
    finally:
        cur.close()


def reload(conn, target_table: str | None) -> bool:
    """Reload all tables (or just one if --table given)."""
    print("=" * 64)
    print("RELOAD MODE - TRUNCATE + PUT + COPY INTO")
    print("=" * 64)

    targets = [
        (csv_rel, t, exp_rows)
        for (csv_rel, t, exp_rows, _) in TABLES
        if (target_table is None or t == target_table)
    ]
    if not targets:
        print(f"ERROR: --table {target_table!r} not in known tables.")
        print("       Valid: TRAIN_MATRIX | DISEASE_DRUG | HOSPITALS")
        return False

    all_ok = True
    for csv_rel, table, exp_rows in targets:
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            print(f"[FAIL] {table}: source CSV not found at {csv_path}")
            all_ok = False
            continue
        if not reload_table(conn, table, csv_path, exp_rows):
            all_ok = False

    print("\n" + "=" * 64)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    print("=" * 64)
    return all_ok


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snowflake migration / verify tool for DADS5001",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/migrate_to_snowflake.py --verify\n"
            "  python scripts/migrate_to_snowflake.py --reload --confirm\n"
            "  python scripts/migrate_to_snowflake.py --reload --table TRAIN_MATRIX --confirm\n"
        ),
    )
    parser.add_argument("--verify", action="store_true",
                        help="Read-only check of row + column counts")
    parser.add_argument("--reload", action="store_true",
                        help="TRUNCATE and re-upload CSVs (destructive)")
    parser.add_argument("--table", choices=[t for _, t, _, _ in TABLES],
                        help="Limit --reload to a single table")
    parser.add_argument("--confirm", action="store_true",
                        help="Required safety flag for --reload")
    args = parser.parse_args()

    if not (args.verify or args.reload):
        parser.print_help()
        sys.exit(1)

    if args.reload and not args.confirm:
        print("ERROR: --reload requires --confirm flag (safety guard).")
        print("       This will TRUNCATE existing tables. Add --confirm to proceed.")
        sys.exit(1)

    creds = load_secrets()
    print(
        f"Connecting to Snowflake: "
        f"account={creds['account']}, user={creds['user']}, "
        f"db={creds['database']}.{creds['schema']}, "
        f"warehouse={creds['warehouse']}"
    )

    conn = connect(creds)
    try:
        if args.verify:
            ok = verify(conn)
            sys.exit(0 if ok else 1)
        if args.reload:
            ok = reload(conn, args.table)
            sys.exit(0 if ok else 1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
