"""
Page 1 · Non-AI Mode
รับ symptom จาก checkbox จัดกลุ่มตาม body_system → TF-IDF score → top-3 specialty
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Make utils importable when run from /pages/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import (
    load_symptom_dict,
    load_specialty_mapping,
    get_scoring_artifacts,
    init_session_state,
    render_disclaimer_sidebar,
    load_drug_mapping,
    load_hospital_hint,
    load_hospitals_master,
    load_specialty_keywords,
    render_drug_panel,
    render_hospital_panel,
)
from utils.scoring import predict, score_tfidf, score_bayes, classify_confidence

# Defensive: utils/styling.py เป็นงานของ Kade ที่ยังไม่ได้ push ขึ้น GitHub
# (รอ BG_R1_web.jpg re-sync) — fallback no-op เพื่อไม่ให้ Streamlit Cloud crash
try:
    from utils.styling import inject_global_css
except ImportError:
    def inject_global_css():
        pass

st.set_page_config(page_title="Non-AI Mode", page_icon="🩺", layout="wide")
inject_global_css()
init_session_state()
render_disclaimer_sidebar()


# ---------------------------------------------------------------------------
# Reset callback (must be defined before button uses it)
# ---------------------------------------------------------------------------
def _clear_all_symptoms():
    """Callback: bump reset_counter เพื่อให้ widget keys เปลี่ยน → Streamlit
    treat เป็น widget ใหม่ทั้งหมด · ป้องกัน state เก่าค้าง"""
    st.session_state.reset_counter = st.session_state.get("reset_counter", 0) + 1
    st.session_state.selected_symptoms = []
    # ลบ widget keys เก่าทั้งหมด (defensive — บาง version ของ Streamlit
    # ยังจำ state แม้ key เปลี่ยน)
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("sym_"):
            del st.session_state[k]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Non-AI Mode")
st.markdown("##### ติ๊กอาการที่คุณมี → ระบบใช้ TF-IDF / Bayes แนะนำ Top-3 แผนก")

st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sym_dict = load_symptom_dict()
mapping = load_specialty_mapping()
arts = get_scoring_artifacts()

# Filter to user-facing symptoms only
sym_visible = sym_dict[sym_dict["is_user_facing"] == True].copy()

# ---------------------------------------------------------------------------
# 1️⃣ Symptom checkbox UI — grouped by body_system
# ---------------------------------------------------------------------------
st.markdown("### 1️⃣ เลือกอาการที่คุณมี")
st.caption(
    f"แสดง {len(sym_visible)} อาการ จัดกลุ่มตามระบบร่างกาย · "
    "พิมพ์ค้นหาด้านบน หรือกางกล่องด้านล่างเพื่อติ๊ก"
)

body_order = (
    sym_visible.groupby("body_system").size()
    .sort_values(ascending=False).index.tolist()
)

# Version counter — bumped by clear button to invalidate widget keys
_v = st.session_state.get("reset_counter", 0)

# ---------------------------------------------------------------------------
# 🔍 Search box (Phase 4 UX) — quick filter ข้าม body_system
# Key prefix `sym_search_*` เพื่อให้ clear button (ที่ลบ key ขึ้นต้น `sym_`) reset ครบ
# ---------------------------------------------------------------------------
_search_term = st.text_input(
    "🔍 ค้นหาอาการ (พิมพ์ภาษาไทยหรืออังกฤษ)",
    placeholder="เช่น: ปวด, ไข้, fever, headache, คัน",
    key=f"sym_search_q_v{_v}",
    help="ระบบจะค้นจาก: ชื่อไทย · คำเรียกอื่น · ชื่ออังกฤษ · label · "
         "ผลการค้นหาจะติ๊กได้เลย sync กับกล่องด้านล่าง",
)

selected = []
_search_shown: set[str] = set()  # symptoms ที่ขึ้นใน search results — กัน double-append

if _search_term and _search_term.strip():
    _q = _search_term.strip().lower()
    _matches = sym_visible[
        sym_visible["symptom_th"].fillna("").str.lower().str.contains(_q, na=False, regex=False) |
        sym_visible["symptom_th_alt"].fillna("").str.lower().str.contains(_q, na=False, regex=False) |
        sym_visible["symptom_en"].fillna("").str.lower().str.contains(_q, na=False, regex=False) |
        sym_visible["ui_label"].fillna("").str.lower().str.contains(_q, na=False, regex=False)
    ].sort_values("symptom_th")

    if len(_matches) == 0:
        st.info(
            f"ℹ️ ไม่พบอาการที่มีคำว่า **'{_search_term}'** · "
            "ลองพิมพ์คำอื่น หรือเปิดกล่องด้านล่างเพื่อหาตามหมวด"
        )
    else:
        with st.container(border=True):
            st.markdown(
                f"#### 🔍 ผลการค้นหา · **{len(_matches)} รายการ** match กับ "
                f"\"{_search_term}\""
            )
            _search_cols = st.columns(2)
            for _i, _row in enumerate(_matches.itertuples()):
                _col = _search_cols[_i % 2]
                _checked = _col.checkbox(
                    f"{_row.ui_label}  ·  _หมวด: {_row.body_system}_",
                    key=f"sym_searchresult_{_row.symptom_en}_v{_v}",
                    value=(_row.symptom_en in st.session_state.selected_symptoms),
                )
                if _checked:
                    selected.append(_row.symptom_en)
                    _search_shown.add(_row.symptom_en)
            st.caption(
                "💡 ติ๊กในผลค้นหาด้านบนนี้ก็ได้ · หรือเลื่อนลงไปหากล่องตามหมวดด้านล่าง · "
                "ทั้ง 2 ที่ sync กันอัตโนมัติ"
            )

# Body-system expanders (existing UI · always shown)
for body in body_order:
    sub = sym_visible[sym_visible["body_system"] == body].sort_values("symptom_th")
    with st.expander(f"**{body}** ({len(sub)})", expanded=False):
        cols = st.columns(2)
        for i, row in enumerate(sub.itertuples()):
            col = cols[i % 2]
            checked = col.checkbox(
                row.ui_label,
                key=f"sym_{row.symptom_en}_v{_v}",
                value=(row.symptom_en in st.session_state.selected_symptoms),
            )
            # de-dup: ถ้าติ๊กผ่าน search section ไปแล้ว · skip append ซ้ำ
            if checked and row.symptom_en not in _search_shown:
                selected.append(row.symptom_en)

st.session_state.selected_symptoms = selected

# ---------------------------------------------------------------------------
# 2️⃣ Selected recap (FIXED: ขึ้นก่อนผลลัพธ์ Top-3)
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 2️⃣ อาการที่คุณติ๊ก")

if not selected:
    st.info("👆 ติ๊กอาการอย่างน้อย 1 ข้อด้านบน เพื่อเริ่มวิเคราะห์")
else:
    sel_df = sym_dict[sym_dict["symptom_en"].isin(selected)][
        ["symptom_th", "ui_label", "body_system"]
    ].rename(
        columns={
            "symptom_th": "อาการ (ไทย)",
            "ui_label": "ป้ายในแอป",
            "body_system": "หมวด",
        }
    )
    a, b = st.columns([1, 4])
    a.metric("ติ๊กแล้ว", len(selected))
    b.dataframe(sel_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Stop here if nothing selected — clear button still useful at end
# ---------------------------------------------------------------------------
if not selected:
    st.divider()
    st.button(
        "🔄 ล้างการเลือกทั้งหมด",
        on_click=_clear_all_symptoms,
        help="ล้าง checkbox + reset",
    )
    st.stop()

# ---------------------------------------------------------------------------
# 3️⃣ Scoring controls
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 3️⃣ เลือกวิธีคำนวณ")

c1, c2 = st.columns([3, 1])
with c1:
    method = st.radio(
        "วิธี scoring",
        options=["tfidf", "bayes", "both"],
        format_func={
            "tfidf": "TF-IDF — ถ่วงน้ำหนักด้วยความเฉพาะของอาการ (recommended)",
            "bayes": "Naive Bayes — คำนวณ P(โรค|อาการ)",
            "both": "เปรียบเทียบทั้งสอง",
        }.get,
        key="scoring_method",
        help="TF-IDF ใช้ specificity (อาการหายาก = น้ำหนักสูง) · "
             "Bayes ใช้ทฤษฎีความน่าจะเป็น Naive Bayes",
    )
with c2:
    top_k = st.number_input(
        "Top-K", min_value=1, max_value=10, value=3,
        help="แสดงกี่อันดับแรก",
    )

# ---------------------------------------------------------------------------
# Run scoring
# ---------------------------------------------------------------------------
with st.spinner("กำลังคำนวณ..."):
    ranked = predict(selected, arts, method=method, top_k=top_k)

# Join with specialty mapping
result = ranked.merge(
    mapping,
    left_on="disease",
    right_on="disease_en",
    how="left",
)

# ---------------------------------------------------------------------------
# 4️⃣ Display top-K
# ---------------------------------------------------------------------------
st.divider()

# Engine info badge — แสดงให้เห็นว่ารันด้วย DuckDB SQL จริง
engine = ranked.attrs.get("engine", "?")
elapsed = ranked.attrs.get("scoring_time_ms", 0)
n_rows = ranked.attrs.get("rows_scanned", 0)
st.caption(
    f"⚡ Powered by **{engine}** · scored {n_rows} diseases in **{elapsed:.1f} ms** "
    f"(method: {method.upper()})"
)

st.markdown(f"### 4️⃣ ผลลัพธ์ Top-{top_k}")

# === Phase 2 + 5: Confidence badge (สูง / กลาง / ต่ำ / ต่ำมาก) ===
# คำนวณจาก full ranking (41 โรค) ก่อน truncate ด้วย Top-K
_full_for_conf = score_tfidf(selected, arts) if method != "bayes" else score_bayes(selected, arts)
_conf = classify_confidence(_full_for_conf, n_user_symptoms=len(selected))
_color_map = {"high": "success", "medium": "warning", "low": "error", "very_low": "warning"}
_st_func = getattr(st, _color_map.get(_conf["level"], "info"))
_st_func(
    f"{_conf['emoji']} **{_conf['label']}** — {_conf['reason']}"
)

# Phase 5: Honest fallback banner เมื่อ very_low/low — แสดงก่อน Top-3
if _conf["level"] == "very_low":
    st.warning(
        "⚠️ **ระบบไม่สามารถสรุปได้ชัดเจน** — ข้อมูลอาการที่ระบุยังน้อยเกินไป "
        "ผลด้านล่างเป็นเพียง **reference เท่านั้น** · กรุณาปรึกษาแพทย์ก่อนตัดสินใจ"
    )
    st.caption(
        "💡 ลองติ๊กอาการเพิ่มอย่างน้อย 3 ข้อ (ตำแหน่ง / ความรุนแรง / อาการร่วม) "
        "ระบบจะให้ผลที่แม่นยำกว่านี้"
    )
elif _conf["level"] == "low":
    st.info(
        "ℹ️ **ผลด้านล่างใช้เป็น reference ประกอบการตัดสินใจ** — ระบบยังไม่มั่นใจมาก · "
        "การปรึกษาแพทย์/เภสัชกรยังเป็นทางเลือกที่ปลอดภัยที่สุด"
    )
    st.caption(
        "💡 ลองติ๊กอาการเพิ่มเติม · ระบบจะแม่นขึ้นเมื่อมีข้อมูลมากกว่านี้"
    )

# ---------------------------------------------------------------------------
# Phase 6 real: Province filter (Global · บนสุดของ result section)
# ---------------------------------------------------------------------------
_hosp_master = load_hospitals_master()
_kw_dict = load_specialty_keywords()
_all_provinces = (
    sorted(_hosp_master["province"].dropna().unique().tolist())
    if not _hosp_master.empty else []
)

if "hosp_p_submitted" not in st.session_state:
    st.session_state["hosp_p_submitted"] = []

st.markdown("#### 🏥 เลือกจังหวัดเพื่อดู รพ.แนะนำใต้แต่ละโรค")
_pc1, _pc2 = st.columns([4, 1])
with _pc1:
    st.multiselect(
        "เลือก 1 จังหวัดขึ้นไป (พิมพ์เพื่อ filter)",
        options=_all_provinces,
        placeholder="เช่น เชียงใหม่, ขอนแก่น, กรุงเทพมหานคร",
        key="hosp_p_select",
        help="พิมพ์ชื่อจังหวัดเพื่อค้นหาใน dropdown · เลือกได้หลายจังหวัด",
    )
with _pc2:
    st.write("")
    if st.button(
        "🔍 ค้นหา รพ.",
        use_container_width=True,
        type="primary",
        key="hosp_search_btn",
    ):
        st.session_state["hosp_p_submitted"] = list(
            st.session_state.get("hosp_p_select", [])
        )

_submitted_provinces = st.session_state.get("hosp_p_submitted", [])
if not _submitted_provinces:
    st.info(
        "👆 กรุณาเลือก **1 จังหวัดขึ้นไป** ในช่องด้านบน แล้วกดปุ่ม "
        "**🔍 ค้นหา รพ.** เพื่อให้ระบบแสดงรายชื่อโรงพยาบาลใต้แต่ละโรค"
    )

URGENCY_LABEL = {
    1: ("🟥 1 — Resuscitation (ฉุกเฉินทันที)", "error"),
    2: ("🟧 2 — Emergent (รีบเข้า รพ.)", "warning"),
    3: ("🟨 3 — Urgent (ภายใน 24 ชม.)", "warning"),
    4: ("🟦 4 — Less urgent (ตามนัด)", "info"),
    5: ("🟩 5 — Non-urgent (ไม่เร่งด่วน)", "success"),
}

for i, row in result.iterrows():
    rank = i + 1
    disease_th = row["disease_th"] if pd.notna(row["disease_th"]) else row["disease"]
    primary = row["primary_specialty"] if pd.notna(row["primary_specialty"]) else "—"
    secondary = row["secondary_specialty"] if pd.notna(row["secondary_specialty"]) else "—"
    urg = int(row["urgency_level"]) if pd.notna(row["urgency_level"]) else 5
    icd = row["icd10_code"] if pd.notna(row["icd10_code"]) else "—"
    red_flags = row["red_flags"] if pd.notna(row["red_flags"]) else ""

    urg_label, urg_kind = URGENCY_LABEL[urg]

    with st.container(border=True):
        a, b = st.columns([3, 1])
        with a:
            st.markdown(f"##### #{rank} · {disease_th}")
            st.caption(f"ICD-10: {icd}")
            st.markdown(f"**แผนกหลัก:** {primary}")
            if secondary != "—":
                st.markdown(f"**แผนกรอง:** {secondary}")
        with b:
            getattr(st, urg_kind)(urg_label)

            if method == "tfidf":
                st.metric(
                    "TF-IDF Score",
                    f"{row['primary_score']:.2f}",
                    help="คะแนนจาก specificity weighting · ยิ่งสูงยิ่งตรงโรคนี้",
                )
                cov = row.get("coverage", 0) or 0
                st.caption(
                    f"Coverage: {cov:.0%} · "
                    "อาการที่ติ๊ก match กับโรคนี้กี่ %"
                )

            elif method == "bayes":
                p = float(row["primary_score"])
                st.metric(
                    "ความน่าจะเป็น",
                    f"{p:.1%}",
                    help=(
                        "P(โรคนี้ | อาการที่ติ๊ก) จาก Naive Bayes · "
                        "ตั้ง prior = 1/41 (uniform บน synthetic data)"
                    ),
                )
                st.caption(
                    f"~ {int(p*100)} จาก 100 คนที่มีอาการแบบนี้ "
                    "น่าจะเป็นโรคนี้"
                )

            else:  # both
                st.metric(
                    "Avg rank",
                    f"{row['avg_rank']:.1f}",
                    help="อันดับเฉลี่ยจาก TF-IDF + Bayes · ต่ำ = ดีกว่า",
                )
                st.caption("ค่าน้อย = อยู่อันดับสูงในทั้ง 2 วิธี")

        if red_flags:
            st.warning(
                f"WARNING: Red flags ที่ควรระวัง — ถ้ามีอาการเหล่านี้ให้รีบไป ER ทันที: "
                f"{red_flags}"
            )

        # === Phase 6: Drug + Hospital info panels (real province-filtered) ===
        _drug_df = load_drug_mapping()
        _hospital_hint_df = load_hospital_hint()
        _disease_key = row.get("disease_en") if pd.notna(row.get("disease_en")) else row["disease"]
        render_drug_panel(_disease_key, _drug_df)
        if pd.notna(primary) and primary != "—":
            render_hospital_panel(
                primary,
                _hospital_hint_df,
                hospitals_df=_hosp_master,
                keywords_dict=_kw_dict,
                selected_provinces=_submitted_provinces,
                key_suffix=f"_r{rank}_{_disease_key}",
            )

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("ตารางผลลัพธ์เต็ม"):
    show_cols = [
        "rank", "disease_th", "primary_specialty",
        "urgency_level", "icd10_code",
    ]
    if method == "tfidf":
        show_cols += ["primary_score", "coverage"]
    elif method == "bayes":
        show_cols += ["primary_score"]
    elif method == "both":
        show_cols = [
            "rank", "disease", "tfidf_score",
            "bayes_posterior", "avg_rank",
        ]

    avail = [c for c in show_cols if c in result.columns]
    st.dataframe(result[avail], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Scoring matrix — DuckDB intermediate result (all 41 diseases scored)
# ---------------------------------------------------------------------------
with st.expander("ดู scoring matrix (intermediate result จาก DuckDB)"):
    st.caption(
        "DataFrame ที่ DuckDB คืนกลับมาก่อน sort + truncate ด้วย Top-K · "
        "เห็นทั้ง 41 โรคพร้อมคะแนนเต็ม"
    )
    if method == "tfidf":
        full = score_tfidf(selected, arts)
        st.dataframe(
            full[["disease", "score", "n_matched", "n_disease_symptoms",
                  "coverage", "confidence"]],
            use_container_width=True, hide_index=True,
        )
    elif method == "bayes":
        full = score_bayes(selected, arts)
        st.dataframe(
            full[["disease", "log_posterior", "posterior", "n_matched"]],
            use_container_width=True, hide_index=True,
        )
    else:
        ft = score_tfidf(selected, arts)[["disease", "score", "coverage"]]
        fb = score_bayes(selected, arts)[["disease", "posterior"]]
        full = ft.merge(fb, on="disease", how="outer").rename(
            columns={"score": "tfidf_score", "posterior": "bayes_posterior"}
        )
        st.dataframe(full, use_container_width=True, hide_index=True)

    a, b, c = st.columns(3)
    a.metric("Engine", ranked.attrs.get("engine", "-"))
    b.metric("Time", f"{ranked.attrs.get('scoring_time_ms', 0):.2f} ms")
    c.metric("Rows scanned", ranked.attrs.get("rows_scanned", 0))

st.divider()
st.button(
    "ล้างการเลือกทั้งหมด",
    on_click=_clear_all_symptoms,
    help="ล้าง checkbox + reset เพื่อเริ่มใหม่",
)
