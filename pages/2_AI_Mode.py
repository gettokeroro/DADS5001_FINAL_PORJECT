"""
Page 2 · AI Mode (Phase 4 — under construction)
รับ free-text → LLM extract symptoms → match dataset → narrate Top-3 + reasoning
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar

st.set_page_config(page_title="AI Mode", page_icon="🤖", layout="wide")
init_session_state()
render_disclaimer_sidebar()

st.title("🤖 AI Mode")
st.markdown("##### พิมพ์อาการเป็นภาษาธรรมชาติ → LLM สรุป Top-3 แผนก พร้อมเหตุผล")

st.divider()

st.warning(
    "🚧 **Phase 4 · กำลังพัฒนา**\n\n"
    "หน้านี้ยังไม่พร้อมใช้งาน · กำลังรอการเลือก LLM provider และ prompt design"
)

# Mock-up of what the page will look like
st.markdown("### Mock-up หน้าตา (จริงเร็วๆ นี้)")

with st.container(border=True):
    st.markdown("**1. พิมพ์อาการของคุณ**")
    st.text_area(
        "ตัวอย่าง: '3 วันมานี้ ไอแห้งๆ ตอนกลางคืน เจ็บคอตอนเช้า มีน้ำมูกใส รู้สึกเพลีย'",
        height=120,
        disabled=True,
        placeholder="ยังไม่พร้อมใช้งาน",
    )
    st.button("วิเคราะห์", disabled=True)

with st.expander("📋 สิ่งที่จะทำใน Phase 4"):
    st.markdown(
        """
        - [ ] เลือก LLM provider (Gemini Flash · Claude Haiku · Gemma local)
        - [ ] ตั้งค่า API key ใน `.streamlit/secrets.toml`
        - [ ] เขียน system prompt + few-shot examples
        - [ ] Pipeline: Thai input → LLM extract symptoms → match → narrate
        - [ ] Output JSON schema validation (Pydantic)
        - [ ] Rate limit per session
        - [ ] Fallback ถ้า LLM ล่ม → switch ไป Non-AI mode
        - [ ] Streaming response (token-by-token)
        - [ ] เก็บ query log ไป external DB (Supabase)
        """
    )

st.info(
    "💡 ระหว่างนี้ใช้ **🩺 Non-AI Mode** ที่ sidebar ซ้ายมือเพื่อทดสอบ scoring engine ที่พร้อมแล้ว"
)
