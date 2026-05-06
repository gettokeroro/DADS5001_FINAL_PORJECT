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
import re
from math import log
from urllib.parse import quote_plus
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
    """Load Disease → ICD-10 → Specialty → Urgency mapping (48 rows · Phase 4: +Influenza, +Dehydration, +Pharyngitis, +Conjunctivitis, +Acute bronchitis, +Electrolyte imbalance, +Vitamin deficiency)."""
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
    """Load disease → drug mapping. Prefer v2 (109 drugs · 48 diseases · ED category)
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


# ---------------------------------------------------------------------------
# Phase 6 real data — hospitals master + specialty keywords
# ---------------------------------------------------------------------------
def _strip_html(s) -> str:
    """Remove HTML tags + collapse whitespace · ใช้กับ specialty_note ที่มี <br />"""
    if pd.isna(s):
        return s
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data(show_spinner="Loading hospitals master...")
def load_hospitals_master() -> pd.DataFrame:
    """Load hospitals_thailand.csv (UTF-8) · strip HTML in specialty_note.
    Phase 6 real — ใช้แทน specialty_hospital_hint เมื่อต้องกรองตามจังหวัด."""
    p = DATA / "processed" / "hospitals_thailand.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "specialty_note" in df.columns:
        df["specialty_note"] = df["specialty_note"].apply(_strip_html)
    return df


@st.cache_data(show_spinner="Loading specialty keywords...")
def load_specialty_keywords() -> dict:
    """Load specialty → keyword list (สำหรับ score รพ. ตามความเชี่ยวชาญ).
    Returns: {specialty_name: [kw1, kw2, ...]}"""
    p = DATA / "processed" / "specialty_keywords.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, encoding="utf-8-sig")
    out = {}
    for _, r in df.iterrows():
        kws = [k.strip() for k in str(r.get("keywords", "")).split(";") if k.strip()]
        out[r["specialty"]] = kws
    return out


def _hospital_type_tier(htype: str) -> int:
    """Map hospital_type → tier (4=บิ๊กเฉพาะทาง, 1=คลินิก/รพ.สต.)"""
    if not htype or pd.isna(htype):
        return 1
    h = str(htype)
    if "ศูนย์" in h or "มหาวิทยาลัย" in h:
        return 4
    if "ทั่วไป" in h:
        return 3
    if "ชุมชน" in h:
        return 2
    return 1


def _score_hospital(row, keywords, recommended_types):
    # Composite score: keyword match * 100 + tier * 10 * boost + log(beds+1)
    from math import log
    note = str(row.get("specialty_note", "") or "").lower()
    kw_hits = sum(1 for k in keywords if k.lower() in note) if keywords else 0
    spec_match = min(kw_hits / 3.0, 1.0)

    htype = str(row.get("hospital_type", "") or "")
    tier = _hospital_type_tier(htype)
    rec_boost = 1.5 if any(rt and rt in htype for rt in recommended_types) else 1.0

    beds = row.get("beds", 0) or 0
    try:
        beds = float(beds)
    except (TypeError, ValueError):
        beds = 0
    if pd.isna(beds):
        beds = 0
    return spec_match * 100 + tier * 10 * rec_boost + log(beds + 1)


# ---------------------------------------------------------------------------
# Phase 6 polish — Badge helpers (drug payment + hospital insurance)
# ---------------------------------------------------------------------------
def drug_payment_badge(ed_category) -> dict:
    """Map ED category → payment/access badge.

    Returns:
        {emoji, label, sub_label, kind} where kind ∈ {success, warning, error, info}
    """
    if pd.isna(ed_category) or not str(ed_category).strip():
        return {
            "emoji": "💵",
            "label": "ชำระเอง",
            "sub_label": "นอกบัญชียาหลักแห่งชาติ",
            "kind": "info",
        }
    cat = str(ed_category).strip()
    # ก/ข/ค — เบิกได้ทุกสิทธิ แต่ tier การเข้าถึงต่าง
    tier_map = {
        "ก": "พื้นฐาน — มีทุก รพ./ร้านยา",
        "ข": "ทั่วไป — มี รพ. ส่วนใหญ่",
        "ค": "เฉพาะทาง — รพ. จังหวัดขึ้นไป",
    }
    if cat in tier_map:
        return {
            "emoji": "🟢",
            "label": "เบิกได้ทุกสิทธิ",
            "sub_label": tier_map[cat],
            "kind": "success",
        }
    if cat == "ง":
        return {
            "emoji": "🟡",
            "label": "เบิกแบบมีเงื่อนไข",
            "sub_label": "ต้องแพทย์เฉพาะทางสั่ง",
            "kind": "warning",
        }
    if cat == "จ1":
        return {
            "emoji": "🟠",
            "label": "เคสพิเศษ",
            "sub_label": "เฉพาะโรคที่กำหนด",
            "kind": "warning",
        }
    if cat == "จ2":
        return {
            "emoji": "🔴",
            "label": "ยาแพง · เคสพิเศษ",
            "sub_label": "เข้าโครงการพิเศษ",
            "kind": "error",
        }
    # fallback
    return {
        "emoji": "❔",
        "label": f"บัญชี {cat}",
        "sub_label": "ตรวจสอบกับเภสัชกร",
        "kind": "info",
    }


def hospital_insurance_badge(affiliation) -> dict:
    """Map affiliation → insurance acceptance badge."""
    if pd.isna(affiliation) or not str(affiliation).strip():
        return {
            "emoji": "🏥",
            "label": "ตรวจสอบกับ รพ. โดยตรง",
            "kind": "info",
        }
    aff = str(affiliation)
    # ทหาร / ตำรวจ
    if "กลาโหม" in aff or "ตำรวจ" in aff:
        return {
            "emoji": "🪖",
            "label": "ทหาร/ตำรวจ + ข้าราชการ · ฉุกเฉินรับทุกสิทธิ",
            "kind": "info",
        }
    # เอกชน
    if "เอกชน" in aff:
        return {
            "emoji": "💼",
            "label": "ประกันสังคม · ชำระเอง  (บัตรทองตามที่ รพ. เข้าร่วม)",
            "kind": "warning",
        }
    # รัฐทุกประเภท (สป.สธ., สธ., กทม., มหาวิทยาลัย, ศึกษาธิการ, อปท., สภากาชาด, ฯลฯ)
    state_keywords = [
        "สาธารณสุข", "สป.สธ", "กทม", "มหาวิทยาลัย", "ศึกษาธิการ",
        "อปท", "ปกครองส่วนท้องถิ่น", "สภากาชาด", "พระราชวัง",
        "กรมการแพทย์",
    ]
    if any(k in aff for k in state_keywords):
        return {
            "emoji": "🏛️",
            "label": "บัตรทอง · ประกันสังคม · ข้าราชการ",
            "kind": "success",
        }
    # อื่นๆ (รัฐวิสาหกิจ, ยุติธรรม, คมนาคม, มหาดไทย, อิสระ, ฯลฯ)
    return {
        "emoji": "🏥",
        "label": "ตรวจสอบกับ รพ. โดยตรง",
        "kind": "info",
    }


def google_maps_url(hospital_th, province=None) -> str:
    """Build Google Maps search URL for a hospital · ใส่จังหวัดเพื่อให้แม่นยำ."""
    parts = [str(hospital_th)]
    if province and pd.notna(province):
        parts.append(str(province))
    query = " ".join(parts)
    return f"https://www.google.com/maps/search/{quote_plus(query)}"


def render_drug_panel(disease_en, drug_df):
    drugs = drug_df[drug_df["disease_en"] == disease_en] if not drug_df.empty else drug_df
    if drugs.empty:
        return
    is_v2 = "drug_generic" in drug_df.columns
    drug_name_col = "drug_generic" if is_v2 else "drug_en"
    dose_col = "dose_note" if is_v2 else "dosage_note"
    cat_col = "ed_category" if is_v2 else "nle_status"

    title = "💊 ยาในบัญชียาหลักที่อาจเกี่ยวข้อง ({n} รายการ)".format(n=len(drugs))
    with st.expander(title):
        st.caption(
            "⚠ **ข้อมูลเพื่อการศึกษาเท่านั้น** · "
            "ห้ามซื้อยา/ใช้ยาเอง · "
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
                        if str(d['prescription_tier']).lower() == "strict":
                            st.caption("🔒 ต้องสั่งโดยแพทย์เฉพาะทาง")
                with col2:
                    cat_val = d.get(cat_col)
                    st.markdown(f"บัญชี **{cat_val if pd.notna(cat_val) else '—'}**")
                    if is_v2:
                        badge = drug_payment_badge(cat_val)
                        st.markdown(
                            f"{badge['emoji']} **{badge['label']}**"
                        )
                        st.caption(badge["sub_label"])


def render_hospital_panel(specialty, hint_df, hospitals_df=None, keywords_dict=None,
                          selected_provinces=None, key_suffix="", max_cards=10):
    if hint_df.empty:
        return
    matched = hint_df[hint_df["primary_specialty"] == specialty]
    hint_row = matched.iloc[0] if not matched.empty else None
    recommended_types = []
    if hint_row is not None:
        recommended_types = [t.strip() for t in str(hint_row["recommended_hospital_types"]).split(",")]

    real_mode = (
        hospitals_df is not None
        and not hospitals_df.empty
        and selected_provinces
    )

    if not real_mode:
        if hint_row is None:
            return
        title = f"🏥 ประเภท รพ.ที่เหมาะกับ {specialty}"
        with st.expander(title):
            st.markdown("**ประเภท รพ.แนะนำ:**")
            for t in recommended_types:
                st.markdown(f"- {t}")
            if pd.notna(hint_row.get("note")):
                st.caption(f"💡 {hint_row['note']}")
            if hospitals_df is not None and not hospitals_df.empty:
                st.info(
                    "👆 เลือกจังหวัด + กดปุ่มค้นหา รพ. ด้านบน "
                    "เพื่อดูรายชื่อโรงพยาบาลในจังหวัดของคุณ"
                )
            else:
                st.caption("🚧 Phase 6: ระบบยังไม่กรองตามจังหวัด")
        return

    sub = hospitals_df[hospitals_df["province"].isin(selected_provinces)].copy()
    label = f"🏥 รพ.แนะนำสำหรับ {specialty} · {', '.join(selected_provinces)}"
    if sub.empty:
        with st.expander(label):
            st.warning(
                f"ไม่พบ รพ. ในจังหวัด {', '.join(selected_provinces)} "
                "(ในฐานข้อมูลปัจจุบัน)"
            )
        return

    keywords = (keywords_dict or {}).get(specialty, [])
    sub["_score"] = sub.apply(
        lambda r: _score_hospital(r, keywords, recommended_types), axis=1
    )
    sub = sub.sort_values("_score", ascending=False).head(max_cards)
    n_total = int((hospitals_df["province"].isin(selected_provinces)).sum())
    suffix = (
        f" (แสดง {len(sub)} จาก {n_total} แห่ง · "
        "เรียงตามความเชี่ยวชาญ + ขนาด)"
    )
    with st.expander(label + suffix):
        if hint_row is not None and pd.notna(hint_row.get("note")):
            st.caption(f"💡 {hint_row['note']}")
        for _, r in sub.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{r['hospital_th']}**  \n*{r.get('hospital_en', '')}*")
                    st.caption(
                        f"📍 {r['province']} · "
                        f"{r.get('hospital_type', '—')} · "
                        f"{r.get('affiliation', '—')}"
                    )
                    note = r.get("specialty_note")
                    if pd.notna(note) and str(note).strip():
                        s = str(note)
                        display = s[:200] + ("..." if len(s) > 200 else "")
                        st.markdown(f"🩺 {display}")
                    # Google Maps link — ใส่จังหวัดเพื่อให้ Google resolve แม่นยำ
                    maps_url = google_maps_url(
                        r["hospital_th"], r.get("province")
                    )
                    st.markdown(
                        f"📞 [โทร · ดูแผนที่บน Google Maps]({maps_url})"
                    )
                with c2:
                    # Insurance badge — replaces former 'beds' metric
                    badge = hospital_insurance_badge(r.get("affiliation"))
                    st.markdown(
                        f"{badge['emoji']} **สิทธิที่รับ**"
                    )
                    st.caption(badge["label"])
                    st.caption(f"H Code: {r.get('h_code', '—')}")
