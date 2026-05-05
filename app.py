"""
Symptom-to-Specialty Triage · Home page
DADS5001 Final Project · Streamlit multi-page app
"""
import streamlit as st
from pathlib import Path

from utils.styling import inject_global_css

st.set_page_config(
    page_title="Symptom-to-Specialty Triage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

# ---------------------------------------------------------------------------
# Sidebar disclaimer (visible on every page via Streamlit's multi-page)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.warning(
        "⚠ **Disclaimer**\n\n"
        "เครื่องมือนี้เป็นโปรเจกต์การศึกษา (DADS5001) "
        "**ไม่ใช่คำวินิจฉัยทางการแพทย์** "
        "กรุณาปรึกษาแพทย์จริงเสมอ"
    )
    st.divider()
    st.caption("Built with Streamlit · DuckDB · pandas · LLM")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.title("🩺 Symptom-to-Specialty Triage")
st.markdown(
    "##### ตัวช่วยตัดสินใจว่าอาการของคุณควรปรึกษาแพทย์แผนกไหน · พร้อมเปรียบเทียบโหมด Non-AI vs AI"
)

st.divider()

# ---------------------------------------------------------------------------
# Quick intro + CTA
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### ปัญหาที่เราพยายามแก้

        คนไทยกูเกิ้ลอาการตัวเอง → เจอข้อมูลปลายทางที่ไม่ตรงบริบท →
        ตื่นตระหนกหรือรักษาผิด → ไม่ไปหาหมอจริง หรือไปผิดแผนก

        ### สิ่งที่แอปนี้ทำให้

        1. **แนะนำแผนก/ประเภทแพทย์** ที่ควรไปจากอาการที่ระบุ
        2. **ระดับความเร่งด่วน 1–5** อิง ED Triage MOPH (1 = ฉุกเฉินทันที, 5 = ไม่เร่งด่วน)
        3. **ค่ารักษาประมาณการ** เปรียบเทียบ รพ.รัฐ/เอกชน *(เร็วๆ นี้)*
        4. **เปรียบเทียบ 2 โหมด** Non-AI (rule-based) vs AI (LLM + RAG)

        ### เริ่มอย่างไร

        เลือก **🩺 Non-AI Mode** จาก sidebar ซ้ายมือเพื่อเริ่มต้น
        หรือกด ปุ่มด้านล่างก็ได้
        """
    )

with col2:
    st.markdown(
        """
        ### Quick stats
        """
    )
    a, b = st.columns(2)
    a.metric("โรคที่รองรับ", "41")
    b.metric("อาการที่รับ", "121")
    a.metric("Specialty", "17")
    b.metric("Urgency", "1–5")

    st.markdown("### Persona")
    st.info("👤 คนกังวลอาการตัวเอง · ก่อนตัดสินใจไป รพ.")

st.divider()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
st.markdown("### How it works")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        #### 1️⃣ ระบุอาการ
        - **Non-AI mode:** ติ๊ก checkbox จาก 121 อาการ จัดกลุ่มตามระบบร่างกาย
        - **AI mode:** พิมพ์อาการเป็นภาษาธรรมชาติ ("3 วันมานี้ ไอแห้ง คอเจ็บ...")
        """
    )
with c2:
    st.markdown(
        """
        #### 2️⃣ ระบบประมวลผล
        - **Non-AI:** TF-IDF scoring + DuckDB join เทียบกับ 41 โรค
        - **AI:** LLM แปลภาษา → match dataset → narrate ผลลัพธ์ + reasoning
        """
    )
with c3:
    st.markdown(
        """
        #### 3️⃣ ผลลัพธ์
        - **Top-3 specialty** พร้อมระดับเร่งด่วน
        - **Red-flag warnings** ถ้าตรงกับสัญญาณรุนแรง
        - **เปรียบเทียบ 2 โหมด** ที่หน้า Compare
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
with st.expander("📋 สถานะปัจจุบันของแต่ละหน้า", expanded=False):
    status = {
        "🏠 Home": "✅ พร้อมใช้",
        "🩺 Non-AI Mode": "✅ พร้อมใช้ (TF-IDF + Bayes)",
        "🤖 AI Mode": "🚧 Phase 4 · กำลังพัฒนา",
        "⚖️ Compare": "🚧 Phase 5 · กำลังพัฒนา",
        "📊 Disease Landscape": "🚧 Phase 5 · กำลังพัฒนา",
        "💰 Cost Estimator": "🚧 Phase 5 · กำลังพัฒนา (Feature B-lite)",
        "ℹ️ About": "✅ พร้อมใช้",
    }
    for page, st_str in status.items():
        st.markdown(f"- **{page}** — {st_str}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "DADS5001 Final Project · Symptom-to-Specialty Triage · "
    "[GitHub](https://github.com/gettokeroro/DADS5001_FINAL_PORJECT)"
)
