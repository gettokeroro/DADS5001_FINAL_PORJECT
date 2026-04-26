"""
Streamlit-cached data loaders.

ใช้ทั้ง 3 cache types ที่โจทย์บังคับ:
    @st.cache_data       — สำหรับ DataFrame, dict, list (immutable returns)
    @st.cache_resource   — สำหรับ DB connection / LLM client (singleton)
    st.session_state     — สำหรับ user state ระหว่าง page (เช่น language toggle)

Usage:
    from utils.data_loader import (
        load_itachi_train, load_specialty_mapping, load_symptom_dict,
        get_duckdb_connection, get_scoring_artifacts,
    )
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path resolution — works whether run from project root or from a page
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent  # utils/data_loader.py → project root


DATA = _project_root() / "data"


# ---------------------------------------------------------------------------
# @st.cache_data — DataFrames (re-loaded only if file changes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading itachi training data...")
def load_itachi_train() -> pd.DataFrame:
    """Load disease-symptom binary matrix (4920 × 132+1)."""
    df = pd.read_csv(DATA / "raw" / "itachi_train.csv")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


@st.cache_data(show_spinner="Loading specialty mapping...")
def load_specialty_mapping() -> pd.DataFrame:
    """Load Disease → ICD-10 → Specialty → Urgency mapping (41 rows)."""
    return pd.read_csv(
        DATA / "processed" / "disease_specialty_mapping.csv",
        encoding="utf-8-sig",
    )


@st.cache_data(show_spinner="Loading Thai symptom dictionary...")
def load_symptom_dict() -> pd.DataFrame:
    """Load Thai symptom dictionary (132 rows)."""
    return pd.read_csv(
        DATA / "processed" / "symptom_dictionary_th.csv",
        encoding="utf-8-sig",
    )


@st.cache_data(show_spinner="Loading symptom specificity (IDF)...")
def load_specificity() -> pd.DataFrame:
    """Load precomputed symptom specificity (132 rows)."""
    return pd.read_csv(
        DATA / "processed" / "symptom_specificity.csv",
        encoding="utf-8-sig",
    )


@st.cache_data
def load_disease_symptom_long() -> pd.DataFrame:
    """Long-form (disease, symptom, freq) — 321 rows. Useful for DuckDB joins."""
    return pd.read_csv(DATA / "processed" / "disease_symptom_long.csv")


# ---------------------------------------------------------------------------
# @st.cache_resource — DB connections (singleton across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing DuckDB connection...")
def get_duckdb_connection():
    """Return a DuckDB in-memory connection with all tables loaded.
    Reused across reruns — initialized once per Streamlit session.
    """
    import duckdb
    con = duckdb.connect(database=":memory:")
    # Register CSVs as tables
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS itachi AS
        SELECT * FROM read_csv_auto('{DATA / "raw" / "itachi_train.csv"}', SAMPLE_SIZE=-1)
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS specialty AS
        SELECT * FROM read_csv_auto('{DATA / "processed" / "disease_specialty_mapping.csv"}', SAMPLE_SIZE=-1)
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS symptom_dict AS
        SELECT * FROM read_csv_auto('{DATA / "processed" / "symptom_dictionary_th.csv"}', SAMPLE_SIZE=-1)
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS specificity AS
        SELECT * FROM read_csv_auto('{DATA / "processed" / "symptom_specificity.csv"}', SAMPLE_SIZE=-1)
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS disease_symptom_long AS
        SELECT * FROM read_csv_auto('{DATA / "processed" / "disease_symptom_long.csv"}', SAMPLE_SIZE=-1)
    """)
    return con


@st.cache_resource(show_spinner="Computing scoring artifacts...")
def get_scoring_artifacts():
    """Pre-compute IDF + freq matrix once per session for scoring engine."""
    from utils.scoring import load_artifacts
    return load_artifacts(data_dir=str(DATA))


# ---------------------------------------------------------------------------
# st.session_state helpers — user state across pages
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize session state defaults if not yet set.
    Call at top of every page."""
    defaults = {
        "selected_symptoms": [],          # list[str] — symptom_en keys
        "language": "th",                  # 'th' or 'en' (currently locked to th)
        "scoring_method": "tfidf",         # 'tfidf', 'bayes', or 'both'
        "last_query": None,                # last user query text (AI mode)
        "history": [],                     # list[dict] — query/result log
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_disclaimer_sidebar():
    """Render the medical disclaimer in sidebar — call on every page."""
    with st.sidebar:
        st.warning(
            "⚠ **Disclaimer**\n\n"
            "เครื่องมือนี้เป็นโปรเจกต์การศึกษา (DADS5001) "
            "**ไม่ใช่คำวินิจฉัยทางการแพทย์** "
            "กรุณาปรึกษาแพทย์จริงเสมอ"
        )
