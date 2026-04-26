"""
Page 5 · Cost Estimator (Feature B-lite, Phase 5 — under construction)
ค่ารักษาประมาณการ + เปรียบเทียบ รพ.รัฐ/เอกชน
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar

st.set_page_config(page_title="Cost Estimator", page_icon="💰", layout="wide")
init_session_state()
render_disclaimer_sidebar()

st.title("💰 Cost Estimator")
st.markdown("##### ค่ารักษาประมาณการ — Feature B (lite) ในสิ่งที่โครงการบอกว่าทำเสริม")

st.divider()

st.warning(
    "🚧 **Phase 5 · กำลังพัฒนา** · ยังไม่ได้หา cost data sources "
    "(HITAP, สปสช. DRG, ราคากลาง)"
)

# Mock UI
c1, c2 = st.columns(2)
with c1:
    st.selectbox("เลือกโรค/หัตถการ", ["ยังไม่พร้อม"], disabled=True)
with c2:
    st.selectbox("ประเภท รพ.", ["รัฐ", "เอกชน", "ทั้งหมด"], disabled=True)

c3, c4, c5 = st.columns(3)
c3.metric("ราคา min", "—")
c4.metric("ราคา median", "—")
c5.metric("ราคา max", "—")

with st.expander("📋 สิ่งที่จะทำใน Phase 5 (Feature B)"):
    st.markdown(
        """
        - [ ] หา cost data: HITAP cost-effectiveness reports
        - [ ] DRG weight ของกระทรวงสาธารณสุข
        - [ ] ราคากลาง สปสช.
        - [ ] ราคาที่ รพ.เอกชนประกาศ (พ.ร.บ.ค่ารักษา)
        - [ ] Filter ตาม รพ. + สิทธิประกันที่มี
        - [ ] แสดงช่วงราคา + caveat (ราคาประมาณการ ไม่ใช่ quote จริง)
        """
    )
