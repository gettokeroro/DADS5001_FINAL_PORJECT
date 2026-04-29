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
from utils.scoring import predict

st.set_page_config(page_title="Non-AI Mode", page_icon="🩺", layout="wide")
init_session_state()
render_disclaimer_sidebar()


# ---------------------------------------------------------------------------
# Reset callback (must be defined before button uses it)
# ---------------------------------------------------------------------------
def _clear_all_symptoms():
    """Callback: ล้าง widget keys + selected_symptoms · ทำงานก่อน rerun
    ห้ามแก้ session_state ของ widget หลัง render — ต้องใช้ callback"""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("sym_"):
            del st.session_state[k]
    st.session_state.selected_symptoms = []


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
    "กางกล่องที่ต้องการเพื่อติ๊ก"
)

body_order = (
    sym_visible.groupby("body_system").size()
    .sort_values(ascending=False).index.tolist()
)
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
st.markdown(f"### 4️⃣ ผลลัพธ์ Top-{top_k}")

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
                    f"≈ {int(p*100)} จาก 100 คนที่มีอาการแบบนี้ "
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
                f"⚠ **Red flags ที่ควรระวัง — ถ้ามีอาการเหล่านี้ให้รีบไป ER ทันที:** "
                f"{red_flags}"
            )

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("📋 ตารางผลลัพธ์เต็ม"):
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
# Reset (FIXED: ใช้ on_click callback ป้องกัน StreamlitAPIException)
# ---------------------------------------------------------------------------
st.divider()
st.button(
    "🔄 ล้างการเลือกทั้งหมด",
    on_click=_clear_all_symptoms,
    help="ล้าง checkbox + reset เพื่อเริ่มใหม่",
)
