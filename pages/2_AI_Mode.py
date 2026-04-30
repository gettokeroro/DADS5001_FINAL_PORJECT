"""
Page 2 · AI Mode (Phase 4 — live)
รับ free-text → Gemini extract symptoms → DuckDB scoring → Gemini narrate
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import (
    load_symptom_dict,
    load_specialty_mapping,
    get_scoring_artifacts,
    init_session_state,
    render_disclaimer_sidebar,
)
from utils.ai_engine import (
    full_pipeline,
    check_rate_limit,
    reset_rate_limit,
    list_available_models,
    _resolve_model_name,
)

st.set_page_config(page_title="AI Mode", page_icon="🤖", layout="wide")
init_session_state()
render_disclaimer_sidebar()

MAX_CALLS_PER_SESSION = 20

# ---------------------------------------------------------------------------
# API key — read from Streamlit secrets
# ---------------------------------------------------------------------------
def _get_api_key():
    """Load Gemini API key from secrets. Returns None if not configured."""
    try:
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 AI Mode")
st.markdown(
    "##### พิมพ์อาการเป็นภาษาธรรมชาติ → Gemini สรุป Top-3 แผนก พร้อมเหตุผล"
)
st.caption(
    "Pipeline: ข้อความไทย → AI extract symptoms → DuckDB scoring → AI narrate · "
    "ใช้ scoring engine เดียวกับ Non-AI Mode"
)

st.divider()

# ---------------------------------------------------------------------------
# Pre-flight: API key check
# ---------------------------------------------------------------------------
api_key = _get_api_key()
if not api_key:
    st.error(
        "⚠ **ยังไม่ได้ตั้งค่า GOOGLE_API_KEY**\n\n"
        "วิธีตั้ง:\n"
        "1. ขอ key ที่ <https://aistudio.google.com/app/apikey> (ฟรี)\n"
        "2. **บน Streamlit Cloud:** Manage app → Settings → Secrets → "
        "วาง `GOOGLE_API_KEY = \"...\"`\n"
        "3. **Local dev:** สร้างไฟล์ `.streamlit/secrets.toml` ใส่ `GOOGLE_API_KEY = \"...\"`\n\n"
        "ระหว่างนี้ลองใช้ **🩺 Non-AI Mode** ที่ sidebar ซ้าย"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Rate limit display
# ---------------------------------------------------------------------------
calls_used = st.session_state.get("ai_call_counter", 0)
calls_left = MAX_CALLS_PER_SESSION - calls_used
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown("### 1️⃣ พิมพ์อาการของคุณเป็นภาษาธรรมชาติ")
with c2:
    st.metric("Calls ใช้ไปแล้ว", f"{calls_used}/{MAX_CALLS_PER_SESSION}")
with c3:
    if st.button("🔄 Reset counter", help="debug — ใช้ตอน present ถ้าหมด"):
        reset_rate_limit(st.session_state)
        st.rerun()

# ---------------------------------------------------------------------------
# 1️⃣ Input
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "3 วันมานี้ ไอแห้งๆ ตอนกลางคืน เจ็บคอตอนเช้า มีน้ำมูกใส รู้สึกเพลีย",
    "ปวดหัวข้างเดียวมา 2 วัน คลื่นไส้ มองภาพเป็นจุดดำ",
    "ไอเป็นเลือด น้ำหนักลดไป 5 กิโล ไข้ตอนกลางคืน เหงื่อท่วม",
    "ฉี่บ่อยมาก กระหายน้ำตลอด น้ำหนักลด ตามัว",
    "เจ็บแน่นหน้าอก ร้าวไปแขนซ้าย เหงื่อท่วม หายใจไม่ออก (ฉุกเฉิน!)",
]

# Use one consistent key for the textarea — example buttons write to it directly
TEXTAREA_KEY = "ai_input_widget"

with st.expander("💡 ตัวอย่างคำถาม (คลิกเพื่อใช้)"):
    for ex in EXAMPLE_QUERIES:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            # เขียนตรงไปที่ widget key — Streamlit จะใช้ตอน render รอบหน้า
            st.session_state[TEXTAREA_KEY] = ex
            st.rerun()

user_text = st.text_area(
    "อธิบายอาการของคุณ",
    height=140,
    placeholder="เช่น '3 วันมานี้ ไอแห้งๆ ตอนกลางคืน เจ็บคอตอนเช้า มีน้ำมูกใส'",
    help="พิมพ์เป็นประโยคหรือรายการอาการ · ระบุระยะเวลา/ความรุนแรงได้เพื่อความแม่นยำ",
    key=TEXTAREA_KEY,
)

go = st.button("🤖 วิเคราะห์ด้วย AI", type="primary", use_container_width=True)

if not go:
    st.info("👆 พิมพ์อาการของคุณด้านบน แล้วกด \"วิเคราะห์ด้วย AI\"")
    st.stop()

if not user_text.strip():
    st.warning("กรุณาพิมพ์อาการก่อนวิเคราะห์")
    st.stop()

# ---------------------------------------------------------------------------
# Rate limit check
# ---------------------------------------------------------------------------
allowed, n = check_rate_limit(st.session_state, max_calls=MAX_CALLS_PER_SESSION)
if not allowed:
    st.error(
        f"⚠ Rate limit reached ({MAX_CALLS_PER_SESSION} calls/session)\n\n"
        "กดปุ่ม **🔄 Reset counter** ด้านบน หรือ refresh หน้านี้เพื่อเริ่มใหม่"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
sym_dict = load_symptom_dict()
mapping = load_specialty_mapping()
arts = get_scoring_artifacts()

with st.spinner("🤖 AI กำลังวิเคราะห์..."):
    result = full_pipeline(
        user_text=user_text,
        dictionary_df=sym_dict[sym_dict["is_user_facing"] == True],
        scoring_arts=arts,
        mapping_df=mapping,
        api_key=api_key,
        method="tfidf",
        top_k=3,
    )

# (no clear — keep textarea content so user สามารถ tweak แล้วลองใหม่)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
if result.error:
    st.error(f"❌ AI Pipeline error: {result.error}")

    # Auto-debug: ถ้าเป็น 404 model not found → แสดง available models
    if "404" in result.error or "not found" in result.error.lower():
        with st.expander("🔧 Debug — Available models บน account นี้", expanded=True):
            models = list_available_models(api_key)
            st.write("Models ที่ใช้ได้กับ key นี้:")
            for m in models:
                st.code(m, language="text")
            st.caption(
                "ถ้าไม่เห็น `gemini-2.5-flash` หรือ `gemini-2.0-flash` "
                "ให้ตรวจ key + permissions ที่ aistudio.google.com"
            )

    st.info(
        "💡 ลองใช้ **🩺 Non-AI Mode** เป็นทางเลือก · หรือลองพิมพ์อาการให้ชัดเจนขึ้น"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline performance badge
# ---------------------------------------------------------------------------
total_ms = result.extract_time_ms + result.scoring_time_ms + result.narrate_time_ms
try:
    used_model = _resolve_model_name(api_key)
except Exception:
    used_model = "auto"
st.caption(
    f"⚡ Pipeline: extract **{result.extract_time_ms:.0f}ms** + "
    f"score **{result.scoring_time_ms:.0f}ms** + "
    f"narrate **{result.narrate_time_ms:.0f}ms** = "
    f"**{total_ms:.0f}ms** total · model: **{used_model}**"
)

st.divider()

# ---------------------------------------------------------------------------
# 2️⃣ Extracted symptoms (transparency)
# ---------------------------------------------------------------------------
st.markdown("### 2️⃣ อาการที่ AI สกัดได้จากข้อความของคุณ")

if result.extracted.symptoms:
    extracted_rows = []
    for s in result.extracted.symptoms:
        # Look up Thai label
        match = sym_dict[sym_dict["symptom_en"] == s.symptom_en]
        th = match["symptom_th"].iloc[0] if len(match) else "(ไม่พบในพจนานุกรม)"
        extracted_rows.append({
            "อาการ (ไทย)": th,
            "code": s.symptom_en,
            "ความมั่นใจ": f"{s.confidence:.0%}",
            "วลีจากคุณ": s.user_phrase or "—",
        })
    st.dataframe(pd.DataFrame(extracted_rows), use_container_width=True, hide_index=True)

    extra = []
    if result.extracted.duration_days:
        extra.append(f"⏰ ระยะเวลาอาการ: {result.extracted.duration_days} วัน")
    if result.extracted.notes:
        extra.append(f"📝 หมายเหตุเพิ่มเติม: {result.extracted.notes}")
    for e in extra:
        st.caption(e)
else:
    st.warning("AI ไม่สามารถสกัดอาการที่ตรงกับฐานข้อมูลได้")

st.divider()

# ---------------------------------------------------------------------------
# 3️⃣ AI narration (markdown)
# ---------------------------------------------------------------------------
st.markdown("### 3️⃣ คำอธิบายจาก AI")
with st.container(border=True):
    st.markdown(result.narration)

# ---------------------------------------------------------------------------
# 4️⃣ Top-3 cards (เหมือน Non-AI mode)
# ---------------------------------------------------------------------------
if not result.ranked_df.empty:
    st.divider()
    st.markdown("### 4️⃣ ผลลัพธ์ Top-3 (จาก scoring engine)")

    enriched = result.ranked_df.merge(
        mapping, left_on="disease", right_on="disease_en", how="left"
    )

    URGENCY_LABEL = {
        1: ("🟥 1 — Resuscitation (ฉุกเฉินทันที)", "error"),
        2: ("🟧 2 — Emergent (รีบเข้า รพ.)", "warning"),
        3: ("🟨 3 — Urgent (ภายใน 24 ชม.)", "warning"),
        4: ("🟦 4 — Less urgent (ตามนัด)", "info"),
        5: ("🟩 5 — Non-urgent (ไม่เร่งด่วน)", "success"),
    }

    for i, row in enriched.iterrows():
        rank = i + 1
        disease_th = row["disease_th"] if pd.notna(row["disease_th"]) else row["disease"]
        primary = row.get("primary_specialty") or "—"
        urg = int(row["urgency_level"]) if pd.notna(row.get("urgency_level")) else 5
        red_flags = row.get("red_flags") or ""
        urg_label, urg_kind = URGENCY_LABEL[urg]

        with st.container(border=True):
            a, b = st.columns([3, 1])
            with a:
                st.markdown(f"##### #{rank} · {disease_th}")
                st.caption(f"ICD-10: {row.get('icd10_code', '—')}")
                st.markdown(f"**แผนกหลัก:** {primary}")
            with b:
                getattr(st, urg_kind)(urg_label)
            if red_flags:
                st.warning(f"⚠ Red flags: {red_flags}")

# ---------------------------------------------------------------------------
# Debug expander — raw JSON + scoring matrix
# ---------------------------------------------------------------------------
with st.expander("🔧 รายละเอียดทางเทคนิค (debug)"):
    tab1, tab2 = st.tabs(["Extracted (JSON)", "Scoring matrix"])
    with tab1:
        st.json(result.extracted.model_dump())
    with tab2:
        if not result.ranked_df.empty:
            cols_to_show = [c for c in
                            ["disease", "primary_score", "n_matched", "coverage", "confidence"]
                            if c in result.ranked_df.columns]
            st.dataframe(result.ranked_df[cols_to_show], use_container_width=True, hide_index=True)
        st.caption(
            f"Engine: {result.ranked_df.attrs.get('engine', '—') if hasattr(result.ranked_df, 'attrs') else '—'}"
        )
