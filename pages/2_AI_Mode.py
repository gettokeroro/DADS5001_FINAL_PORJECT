"""
Page 2 · AI Mode (Phase 4 + 1.5 — interactive follow-up)
รับ free-text → Gemini extract → ถามอาการเพิ่มถ้าน้อยกว่า 4 → DuckDB scoring → Gemini narrate
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
    extract_symptoms,
    narrate_result,
    check_rate_limit,
    reset_rate_limit,
    list_available_models,
    _resolve_model_name,
    AIResult,
    ExtractedSymptoms,
)
from utils.scoring import predict, score_tfidf, score_bayes, suggest_co_symptoms

st.set_page_config(page_title="AI Mode", page_icon="🤖", layout="wide")
init_session_state()
render_disclaimer_sidebar()

MAX_CALLS_PER_SESSION = 20
FOLLOWUP_THRESHOLD = 4   # ถ้า extracted < นี้ → ถามเพิ่ม
TOP_K_CO_SYMPTOMS = 5


def _get_api_key():
    try:
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Initialize state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "ai_step": "input",                # input | followup | result
    "ai_extracted": None,              # ExtractedSymptoms
    "ai_initial_text": "",
    "ai_co_symptoms_df": None,         # DataFrame
    "ai_extra_codes": [],              # list[str] selected in followup
    "ai_final_result": None,           # AIResult
    "ai_call_counter": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

TEXTAREA_KEY = "ai_input_widget"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 AI Mode")
st.markdown("##### พิมพ์อาการเป็นภาษาธรรมชาติ → น้องอุ่นในจะถามเพิ่มถ้าข้อมูลน้อย → Top-3 แผนก")

# Rate limit
calls_used = st.session_state.get("ai_call_counter", 0)
calls_left = MAX_CALLS_PER_SESSION - calls_used
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    pass
with c2:
    st.metric("Calls", f"{calls_used}/{MAX_CALLS_PER_SESSION}")
with c3:
    if st.button("🔄 Reset", help="reset counter + state"):
        for k in list(DEFAULTS.keys()):
            st.session_state[k] = DEFAULTS[k]
        # Force textarea fresh
        if TEXTAREA_KEY in st.session_state:
            del st.session_state[TEXTAREA_KEY]
        st.rerun()

api_key = _get_api_key()
if not api_key:
    st.error(
        "⚠ ยังไม่ได้ตั้งค่า GOOGLE_API_KEY · "
        "ขอ key ที่ aistudio.google.com แล้วใส่ใน Streamlit Cloud → Settings → Secrets"
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
sym_dict = load_symptom_dict()
mapping = load_specialty_mapping()
arts = get_scoring_artifacts()
visible_dict = sym_dict[sym_dict["is_user_facing"] == True]


# ===========================================================================
# STEP: input — กรอกอาการครั้งแรก
# ===========================================================================
if st.session_state.ai_step == "input":
    st.markdown("### 1️⃣ พิมพ์อาการของคุณเป็นภาษาธรรมชาติ")

    EXAMPLE_QUERIES = [
        "3 วันมานี้ ไอแห้งๆ ตอนกลางคืน เจ็บคอตอนเช้า",
        "ปวดหัวข้างเดียวมา 2 วัน คลื่นไส้ มองภาพเป็นจุดดำ",
        "รู้สึกเหนื่อยง่ายขึ้น ไม่อยากกินอาหาร",
        "ฉี่บ่อยมาก กระหายน้ำตลอด น้ำหนักลด",
        "เจ็บแน่นหน้าอก ร้าวไปแขนซ้าย เหงื่อท่วม (ฉุกเฉิน!)",
    ]
    with st.expander("💡 ตัวอย่างคำถาม (คลิกเพื่อใช้)"):
        for ex in EXAMPLE_QUERIES:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state[TEXTAREA_KEY] = ex
                st.rerun()

    st.text_area(
        "อธิบายอาการของคุณ",
        height=140,
        placeholder="เช่น 'ไอแห้งๆ 3 วัน เจ็บคอ มีน้ำมูก'",
        key=TEXTAREA_KEY,
    )

    user_text = st.session_state.get(TEXTAREA_KEY, "")
    if st.button("🤖 วิเคราะห์ด้วย AI", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("กรุณาพิมพ์อาการก่อน")
            st.stop()

        # Rate limit (1 call สำหรับ extract)
        allowed, _ = check_rate_limit(st.session_state, max_calls=MAX_CALLS_PER_SESSION)
        if not allowed:
            st.error(f"Rate limit ถึงแล้ว ({MAX_CALLS_PER_SESSION} calls/session)")
            st.stop()

        with st.spinner("🤖 น้องอุ่นในกำลังอ่านอาการของพี่..."):
            try:
                extracted, _t = extract_symptoms(user_text, visible_dict, api_key)
            except ValueError as e:
                st.error(f"❌ Extract error: {e}")
                if "404" in str(e) or "not found" in str(e).lower():
                    with st.expander("🔧 Available models"):
                        for m in list_available_models(api_key):
                            st.code(m)
                st.stop()

        st.session_state.ai_initial_text = user_text
        st.session_state.ai_extracted = extracted

        # Decide next step: followup if extracted < threshold
        n_extracted = len(extracted.symptoms)
        if n_extracted < FOLLOWUP_THRESHOLD:
            # Compute co-symptoms (no AI call needed — pure SQL)
            initial_codes = [s.symptom_en for s in extracted.symptoms]
            co_df = suggest_co_symptoms(initial_codes, arts, top_k=TOP_K_CO_SYMPTOMS)
            # Join with Thai labels
            co_df = co_df.merge(
                visible_dict[["symptom_en", "symptom_th", "ui_label", "body_system"]],
                left_on="symptom", right_on="symptom_en", how="left"
            )
            st.session_state.ai_co_symptoms_df = co_df
            st.session_state.ai_step = "followup"
        else:
            st.session_state.ai_step = "result"
        st.rerun()


# ===========================================================================
# STEP: followup — ถามอาการเพิ่ม
# ===========================================================================
elif st.session_state.ai_step == "followup":
    extracted = st.session_state.ai_extracted
    co_df = st.session_state.ai_co_symptoms_df

    st.markdown("### 2️⃣ น้องอุ่นในเข้าใจอาการของพี่แล้ว")

    # Show what was extracted
    if extracted.symptoms:
        rows = []
        for s in extracted.symptoms:
            match = sym_dict[sym_dict["symptom_en"] == s.symptom_en]
            th = match["symptom_th"].iloc[0] if len(match) else s.symptom_en
            rows.append({"อาการ": th, "code": s.symptom_en, "ความมั่นใจ": f"{s.confidence:.0%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("น้องยังจับอาการของพี่ไม่ได้ ลองพิมพ์ใหม่ที่ชัดเจนขึ้นนะครับ")

    st.divider()
    st.markdown("### 3️⃣ ขออนุญาตถามเพิ่มอีกนิดนะครับ 🙏")
    st.caption(
        "อาการที่พี่บอกมายังน้อยอยู่ · "
        "น้องลองหาอาการที่มัก **เกิดร่วมกัน** กับสิ่งที่พี่บอกมา · "
        "ติ๊กที่ตรงกับที่พี่รู้สึกด้วย เพื่อให้ผลแม่นขึ้น"
    )

    # Checkboxes for co-symptoms
    if not co_df.empty:
        extra_selected = []
        cols = st.columns(2)
        for i, row in enumerate(co_df.itertuples()):
            col = cols[i % 2]
            label = row.ui_label if pd.notna(row.ui_label) else row.symptom
            checked = col.checkbox(
                f"{label}",
                key=f"extra_{row.symptom}",
                value=False,
                help=f"พบใน {row.n_diseases_have_it} โรคที่ใกล้เคียงกับอาการที่พี่บอก",
            )
            if checked:
                extra_selected.append(row.symptom)
        st.session_state.ai_extra_codes = extra_selected

    st.divider()
    a, b = st.columns(2)
    with a:
        if st.button("✅ วิเคราะห์ต่อ (รวมอาการทั้งหมด)", type="primary", use_container_width=True):
            st.session_state.ai_step = "result"
            st.rerun()
    with b:
        if st.button("⏭ ข้ามไปวิเคราะห์เลย (ใช้แค่อาการเดิม)", use_container_width=True):
            st.session_state.ai_extra_codes = []
            st.session_state.ai_step = "result"
            st.rerun()


# ===========================================================================
# STEP: result — รัน scoring + narrate
# ===========================================================================
elif st.session_state.ai_step == "result":
    extracted = st.session_state.ai_extracted
    extra_codes = st.session_state.ai_extra_codes
    initial_codes = [s.symptom_en for s in extracted.symptoms]
    all_codes = list(dict.fromkeys(initial_codes + extra_codes))  # dedupe, preserve order

    if not all_codes:
        st.warning("ไม่มีอาการให้วิเคราะห์ · ลองใหม่")
        if st.button("← กลับไปแก้ไข"):
            st.session_state.ai_step = "input"
            st.rerun()
        st.stop()

    # Run scoring + narrate
    with st.spinner("🤖 น้องอุ่นในกำลังวิเคราะห์..."):
        ranked = predict(all_codes, arts, method="tfidf", top_k=3)
        # Rate limit for narrate
        allowed, _ = check_rate_limit(st.session_state, max_calls=MAX_CALLS_PER_SESSION)
        if not allowed:
            st.error(f"Rate limit ถึงแล้ว ({MAX_CALLS_PER_SESSION} calls/session)")
            st.stop()
        try:
            narration, t_narr = narrate_result(
                st.session_state.ai_initial_text, ranked, mapping, api_key
            )
        except ValueError as e:
            st.error(f"❌ Narrate error: {e}")
            st.stop()

    # Performance badge
    try:
        used_model = _resolve_model_name(api_key)
    except Exception:
        used_model = "auto"
    score_ms = ranked.attrs.get("scoring_time_ms", 0)
    st.caption(
        f"⚡ Symptoms: {len(initial_codes)} จากการพิมพ์ + {len(extra_codes)} จาก follow-up = **{len(all_codes)} รวม** · "
        f"score {score_ms:.0f}ms + narrate {t_narr:.0f}ms · model: **{used_model}**"
    )

    st.divider()

    # Section 2: extracted + extra
    st.markdown("### 2️⃣ อาการที่ใช้วิเคราะห์ทั้งหมด")
    sel_df = sym_dict[sym_dict["symptom_en"].isin(all_codes)][
        ["symptom_th", "ui_label", "body_system"]
    ].copy()
    sel_df["source"] = sel_df.index.map(
        lambda i: "📝 พี่พิมพ์" if sym_dict.iloc[i]["symptom_en"] in initial_codes else "✅ Follow-up"
    )
    a, b = st.columns([1, 4])
    a.metric("รวม", len(all_codes))
    b.dataframe(
        sel_df.rename(columns={
            "symptom_th": "อาการ", "ui_label": "Label",
            "body_system": "หมวด", "source": "ที่มา",
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Section 3: AI narration
    st.markdown("### 3️⃣ คำอธิบายจาก AI")
    with st.container(border=True):
        st.markdown(narration)

    # Section 4: Top-3 cards
    if not ranked.empty:
        st.divider()
        st.markdown("### 4️⃣ ผลลัพธ์ Top-3 (จาก scoring engine)")
        enriched = ranked.merge(mapping, left_on="disease", right_on="disease_en", how="left")

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

    # Buttons
    st.divider()
    a, b = st.columns(2)
    with a:
        if st.button("🔄 ทดสอบอาการอื่น", type="primary", use_container_width=True):
            for k in list(DEFAULTS.keys()):
                st.session_state[k] = DEFAULTS[k]
            if TEXTAREA_KEY in st.session_state:
                del st.session_state[TEXTAREA_KEY]
            st.rerun()
    with b:
        with st.expander("🔧 รายละเอียดเทคนิค"):
            tab1, tab2 = st.tabs(["Extracted (JSON)", "Scoring matrix"])
            with tab1:
                st.json(extracted.model_dump())
            with tab2:
                cols_show = [c for c in
                             ["disease", "primary_score", "n_matched", "coverage", "confidence"]
                             if c in ranked.columns]
                st.dataframe(ranked[cols_show], use_container_width=True, hide_index=True)
