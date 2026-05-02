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


# ---------------------------------------------------------------------------
# Phase 6 — Drug + Hospital info loaders (skeleton with hard-coded mappings)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading drug mapping...")
def load_drug_mapping() -> pd.DataFrame:
    """Load disease → drug mapping. Prefer v2 (82 drugs · 41 diseases · ED category)
    over skeleton v1 (15 drugs · 11 diseases) if available."""
    v2 = DATA / "processed" / "disease_drug_mapping_v2_ed.csv"
    v1 = DATA / "processed" / "disease_drug_mapping.csv"
    p = v2 if v2.exists() else v1
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner="Loading hospital hint...")
def load_hospital_hint() -> pd.DataFrame:
    """Load specialty → recommended hospital types (skeleton).
    Future: replace with real hospital master data filtered by province."""
    p = DATA / "processed" / "specialty_hospital_hint.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else pd.DataFrame()


def render_drug_panel(disease_en: str, drug_df: pd.DataFrame):
    """Render drug expander for a disease (educational only).
    Compatible with both v1 schema (drug_en, dosage_note, nle_status) and
    v2 schema (drug_generic, dose_note, ed_category, reimbursement_note)."""
    drugs = drug_df[drug_df["disease_en"] == disease_en] if not drug_df.empty else drug_df
    if drugs.empty:
        return
    # Detect schema version
    is_v2 = "drug_generic" in drug_df.columns
    drug_name_col = "drug_generic" if is_v2 else "drug_en"
    dose_col = "dose_note" if is_v2 else "dosage_note"
    cat_col = "ed_category" if is_v2 else "nle_status"

    with st.expander(f"💊 ยาในบัญชียาหลักที่อาจเกี่ยวข้อง ({len(drugs)} รายการ)"):
        st.caption(
            "⚠ **ข้อมูลเพื่อการศึกษาเท่านั้น** · ห้ามซื้อยา/ใช้ยาเอง · "
            "ยาเหล่านี้ต้องได้รับการสั่งโดยแพทย์/เภสัชกรเท่านั้น"
        )
        for _, d in drugs.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{d.get(drug_name_col, '?')}** · {d.get('drug_th', '')}")
                    if pd.notna(d.get('indication_th')):
                        st.caption(f"📋 ข้อบ่งใช้: {d['indication_th']}")
                    if pd.notna(d.get(dose_col)):
                        st.caption(f"📏 ขนาดอ้างอิง: {d[dose_col]}")
                    if is_v2 and pd.notna(d.get('reimbursement_note')):
                        st.caption(f"💰 เบิก: {d['reimbursement_note']}")
                    if is_v2 and pd.notna(d.get('prescription_tier')):
                        tier = d['prescription_tier']
                        if str(tier).lower() == "strict":
                            st.caption(f"🔒 ต้องสั่งโดยแพทย์เฉพาะทาง")
                with col2:
                    nle = d.get(cat_col, '?')
                    st.markdown(f"บัญชี **{nle}**")


def render_hospital_panel(specialty: str, hint_df: pd.DataFrame):
    """Render hospital type recommendation expander (skeleton)."""
    if hint_df.empty:
        return
    matched = hint_df[hint_df["primary_specialty"] == specialty]
    if matched.empty:
        return
    row = matched.iloc[0]
    with st.expander(f"🏥 ประเภท รพ.ที่เหมาะกับ {specialty}"):
        types = str(row["recommended_hospital_types"]).split(",")
        st.markdown("**ประเภท รพ.แนะนำ:**")
        for t in types:
            st.markdown(f"- {t.strip()}")
        if pd.notna(row.get("note")):
            st.caption(f"💡 {row['note']}")
        st.caption(
            "🚧 Phase 6: ระบบยังไม่กรองตามจังหวัด · "
            "สัปดาห์หน้าจะเพิ่ม province filter จาก data.go.th hospital master"
        )
