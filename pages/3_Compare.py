"""
Page 3 · Compare Non-AI vs AI (Phase 5 — under construction)
Side-by-side ผลของ 2 modes บนอาการเดียวกัน · พระเอกของโปรเจกต์
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar
from utils.styling import inject_global_css

st.set_page_config(page_title="Compare", page_icon="⚖️", layout="wide")
inject_global_css()
init_session_state()
render_disclaimer_sidebar()

st.title("⚖️ Compare · Non-AI vs AI")
st.markdown("##### หน้าหลักของการนำเสนอ — แสดงให้เห็นว่า AI mode เพิ่มอะไรจาก Non-AI")

st.divider()

st.warning("🚧 **Phase 5 · กำลังพัฒนา** · พร้อมใช้งานหลัง AI mode (Phase 4) เสร็จ")

# Mock-up
left, right = st.columns(2)

with left:
    st.markdown("### 🩺 Non-AI Result")
    with st.container(border=True):
        st.caption("Top-3 (TF-IDF score)")
        st.markdown("1. _ผลลัพธ์ที่จะแสดง_")
        st.markdown("2. _ผลลัพธ์ที่จะแสดง_")
        st.markdown("3. _ผลลัพธ์ที่จะแสดง_")
        st.caption("⏱ รอบ Run: ~50 ms · ฟรี")

with right:
    st.markdown("### 🤖 AI Result")
    with st.container(border=True):
        st.caption("Top-3 (LLM + RAG)")
        st.markdown("1. _ผลลัพธ์ที่จะแสดง + reasoning_")
        st.markdown("2. _ผลลัพธ์ที่จะแสดง + reasoning_")
        st.markdown("3. _ผลลัพธ์ที่จะแสดง + reasoning_")
        st.caption("⏱ รอบ Run: ~2-5 s · token cost")

st.divider()

st.markdown("### Metrics ที่เปรียบเทียบ")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Top-1 hit rate", "—", help="ความถูกต้อง top-1 บน eval set")
m2.metric("Top-3 hit rate", "—", help="ความถูกต้อง top-3 บน eval set")
m3.metric("Avg latency", "—", help="เวลาตอบสนองเฉลี่ย")
m4.metric("Cost / query", "—", help="ค่า LLM ต่อ 1 query")

with st.expander("📋 สิ่งที่จะทำใน Phase 5"):
    st.markdown(
        """
        - [ ] รับ user query (text หรือ checkbox) ครั้งเดียว
        - [ ] ส่งไปทั้ง 2 modes ในเวลาเดียวกัน
        - [ ] แสดงผลข้างๆ กัน
        - [ ] Diff highlight: AI mode พบอะไรเพิ่มที่ Non-AI ไม่เห็น
        - [ ] Eval comparison: Top-1/Top-3 hit rate · latency · cost
        - [ ] ปุ่ม "Try with another case" ดึง eval case จาก scoring.DEFAULT_EVAL_CASES
        """
    )
