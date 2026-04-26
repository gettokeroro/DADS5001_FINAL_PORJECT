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
)
from utils.scoring import predict, score_tfidf, score_bayes

st.set_page_config(page_title="Non-AI Mode", page_icon="🩺", layout="wide")
init_session_state()
render_disclaimer_sidebar()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Non-AI Mode")
st.markdown("##### ติ๊กอาการที่คุณมี → ระบบใช้ TF-IDF scoring แนะนำ Top-3 แผนก")

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
# Symptom checkbox UI — grouped by body_system
# ---------------------------------------------------------------------------
st.markdown("### 1️⃣ เลือกอาการที่คุณมี")
st.caption(f"แสดง {len(sym_visible)} อาการ จัดกลุ่มตามระบบร่างกาย · กางกล่องที่ต้องการเพื่อติ๊ก")

# Order body systems by # of symptoms desc (likely to be expanded first)
body_order = (
    sym_visible.groupby("body_system").size().sort_values(ascending=False).index.tolist()
)

# Default-expand top 3 most-likely-used groups
DEFAULT_EXPANDED = {"ทั่วไป", "ทางเดินอาหาร", "ทางเดินหายใจ"}

selected = []
for body in body_order:
    sub = sym_visible[sym_visible["body_system"] == body].sort_values("symptom_th")
    expand = body in DEFAULT_EXPANDED
    with st.expander(f"**{body}** ({len(sub)})", expanded=expand):
        cols = st.columns(2)
        for i, row in enumerate(sub.itertuples()):
            col = cols[i % 2]
            checked = col.checkbox(
                row.ui_label,
                key=f"sym_{row.symptom_en}",
                value=(row.symptom_en in st.session_state.selected_symptoms),
            )
            if checked:
                selected.append(row.symptom_en)

st.session_state.selected_symptoms = selected

# ---------------------------------------------------------------------------
# Scoring controls
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 2️⃣ เลือกวิธีคำนวณ และดูผลลัพธ์")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    method = st.radio(
        "วิธี scoring",
        options=["tfidf", "bayes", "both"],
        format_func={
            "tfidf": "TF-IDF (recommended)",
            "bayes": "Naive Bayes",
            "both": "เปรียบเทียบทั้งสอง",
        }.get,
        horizontal=True,
        key="scoring_method",
    )
with c2:
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=3)
with c3:
    st.metric("อาการที่ติ๊ก", len(selected))

# ---------------------------------------------------------------------------
# Run scoring
# ---------------------------------------------------------------------------
if not selected:
    st.info("👆 ติ๊กอาการอย่างน้อย 1 ข้อด้านบน เพื่อเริ่มวิเคราะห์")
    st.stop()

with st.spinner("กำลังคำนวณ..."):
    if method == "both":
        ranked = predict(selected, arts, method="both", top_k=top_k)
        score_col = "avg_rank"
        ascending = True
    else:
        ranked = predict(selected, arts, method=method, top_k=top_k)
        score_col = "primary_score"
        ascending = False

# Join with specialty mapping
result = ranked.merge(
    mapping,
    left_on="disease",
    right_on="disease_en",
    how="left",
)

# ---------------------------------------------------------------------------
# Display top-K
# ---------------------------------------------------------------------------
st.divider()
st.markdown(f"### 3️⃣ ผลลัพธ์ Top-{top_k}")

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
                st.metric("Score", f"{row['primary_score']:.2f}")
                st.caption(f"Coverage: {row.get('coverage', 0):.0%}")
            elif method == "bayes":
                st.metric("Posterior", f"{row['primary_score']:.3f}")
            else:
                st.metric("Avg rank", f"{row['avg_rank']:.1f}")

        if red_flags:
            st.warning(f"⚠ **Red flags ที่ควรระวัง — ถ้ามีอาการเหล่านี้ให้รีบไป ER ทันที:** {red_flags}")

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("📋 ตารางผลลัพธ์เต็ม"):
    show_cols = ["rank", "disease_th", "primary_specialty", "urgency_level", "icd10_code"]
    if method == "tfidf":
        show_cols += ["primary_score", "coverage"]
    elif method == "bayes":
        show_cols += ["primary_score"]
    elif method == "both":
        show_cols = ["rank", "disease", "tfidf_score", "bayes_posterior", "avg_rank"]

    avail = [c for c in show_cols if c in result.columns]
    st.dataframe(result[avail], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Selected symptoms recap
# ---------------------------------------------------------------------------
with st.expander("✓ อาการที่คุณติ๊ก", expanded=False):
    sel_df = sym_dict[sym_dict["symptom_en"].isin(selected)][
        ["symptom_th", "ui_label", "body_system"]
    ]
    st.dataframe(sel_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
if st.button("🔄 ล้างการเลือกทั้งหมด"):
    st.session_state.selected_symptoms = []
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("sym_"):
            st.session_state[k] = False
    st.rerun()
