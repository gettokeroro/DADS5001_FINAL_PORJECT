"""
Page 4 · Disease Landscape (Phase 5 — under construction)
Visualization จาก data.go.th: trend 8 ปี · per-capita map · top diseases
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar

st.set_page_config(page_title="Disease Landscape", page_icon="📊", layout="wide")
init_session_state()
render_disclaimer_sidebar()

st.title("📊 Disease Landscape Thailand")
st.markdown("##### สถิติโรคในประเทศไทยจาก data.go.th — context สำหรับ user ก่อนตัดสินใจ")

st.divider()

st.warning("🚧 **Phase 5 · กำลังพัฒนา** · ต้องดาวน์โหลด dataset จาก data.go.th และ clean ก่อน")

# Mock placeholder
tab1, tab2, tab3, tab4 = st.tabs([
    "Trend 8 ปี",
    "Top OPD/IPD",
    "แผนที่จังหวัด",
    "เฝ้าระวัง"
])

with tab1:
    st.markdown("### Trend top-20 IPD 8 ปี (2559–2566)")
    st.caption("Source: data.go.th — st10-298")
    st.empty()
    st.info("💡 แสดง Plotly line chart ที่นี่ — เลือกโรคได้, hover เห็นค่าจริง")

with tab2:
    st.markdown("### Top diseases · OPD vs IPD")
    st.caption("Source: data.go.th — 64-65001 + 64-65002")
    st.info("💡 แสดง bar chart เปรียบเทียบ OPD vs IPD")

with tab3:
    st.markdown("### Choropleth map ประเทศไทย")
    st.caption("Per-capita disease incidence — normalize ด้วยประชากร 2565")
    st.info("💡 Plotly choropleth + GeoJSON ของจังหวัดไทย")

with tab4:
    st.markdown("### โรคติดต่อ/เฝ้าระวัง 2567")
    st.caption("Source: data.go.th — dataset_20_025")
    st.info("💡 Alert table + map ของจังหวัดที่ระบาด")

st.divider()
with st.expander("📋 สิ่งที่จะทำใน Phase 5"):
    st.markdown(
        """
        **Data ingestion (priority):**
        - [ ] ดาวน์โหลด 6 datasets จาก data.go.th (Tier S + A ที่รีวิวแล้ว)
        - [ ] Encoding handling (utf-8-sig vs cp874)
        - [ ] Save to `data/raw/datago/`
        - [ ] Clean + map column names
        - [ ] DuckDB views

        **Visualizations:**
        - [ ] Time series (Plotly line)
        - [ ] Choropleth (จังหวัด/เขตสุขภาพ)
        - [ ] Bar chart top-20 ranking
        - [ ] Heatmap disease × province × year
        - [ ] Treemap by ICD-10 chapter
        """
    )
