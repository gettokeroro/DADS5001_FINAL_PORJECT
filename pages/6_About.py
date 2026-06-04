"""
Page 6 · About — methodology, data sources, team, license
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar
from utils.styling import inject_global_css

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
inject_global_css()
init_session_state()
render_disclaimer_sidebar()

st.title("ℹ️ About this project")

st.markdown(
    """
    ## ภาพรวม

    **Symptom-to-Specialty Triage** เป็น Final Project ของรายวิชา **DADS5001**
    เป้าหมายคือสร้าง Data-centric Streamlit application ที่ช่วยคนไทยตัดสินใจว่าควรไป
    โรงพยาบาลแผนกไหน + ระดับเร่งด่วนแค่ไหน จากอาการที่ระบุ
    พร้อมเปรียบเทียบโหมด **Non-AI** (rule-based) กับ **AI** (LLM + RAG)

    ## ปัญหาที่แก้

    คนไทยมักกูเกิ้ลอาการตัวเอง → เจอข้อมูลปลายทางที่ไม่ตรงบริบท →
    ตื่นตระหนกหรือรักษาผิด → ไม่ไปหาหมอจริง หรือไปผิดแผนก

    ## ขอบเขต (in / out)

    **In-scope:**
    - แนะนำ specialty (แผนก) จากอาการ
    - ระดับเร่งด่วน 1–5 (อิง ED Triage MOPH)
    - ยาอ้างอิงตามโรค + บัญชียาหลักแห่งชาติ (ED category ก–จ2)
    - โรงพยาบาลใกล้เคียงตามจังหวัด
    - Cost transparency (lite)

    **Out-of-scope (ห้ามทำ):**
    - ❌ วินิจฉัยโรคแน่นอน
    - ❌ สั่งจ่าย/เปลี่ยนยาเอง (ยาที่แสดงเป็นข้อมูลอ้างอิงเท่านั้น)
    - ❌ ทดแทนแพทย์
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
st.markdown("## Methodology")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        ### Non-AI mode
        - **Engine:** TF-IDF scoring + Naive Bayes (เปรียบเทียบ)
        - **Input:** checkbox 121 อาการ จัดกลุ่มตาม body system
        - **Output:** Top-3 disease + specialty + urgency 1-5
        - **Scoring:** TF-IDF บนตาราง disease × symptom freq matrix
        - **Backbone data:** itachi9604 (Kaggle) — augment เป็น 48 disease × 131 symptom
        """
    )
with c2:
    st.markdown(
        """
        ### AI mode (Phase 4)
        - **Pipeline:** Thai free-text → LLM extract → match dataset → narrate
        - **Grounding:** RAG จาก disease description + ICD-10-TM
        - **Guardrail:** input filter, JSON schema validation, rate limit
        - **Output:** top-3 ranked + reasoning + red-flag warnings
        - **Fallback:** ถ้า LLM ล่ม → switch ไป Non-AI mode
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
st.markdown("## Data sources")

st.markdown(
    """
    ### Symptom-Disease backbone
    - **[itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)** — Kaggle · augment เป็น 48 diseases × 131 symptoms (binary)

    ### Thai standards
    - **ICD-10-TM** (กรมการแพทย์ MOPH) — Thai modification ontology
    - **[ED Triage MOPH](https://www.dms.go.th/backend//Content/Content_File/Population_Health/Attach/25621021104459AM_44.pdf?contentId=18326)** — 5-level triage guidelines

    ### Thai disease statistics ([data.go.th](https://www.data.go.th))
    - การจัดอันดับโรคของผู้ป่วยนอก/ใน 2564
    - 20 ลำดับโรคสูงสุด IPD 298 กลุ่มโรค 2559–2566 (time series 8 ปี)
    - ผู้ป่วยนอก 21 กลุ่มโรค 2559–2565
    - การเข้ารับบริการ OPD 2566
    - โรคติดต่อ/เฝ้าระวัง 2567
    - ประชากรตามอายุ สิทธิหลักประกันสุขภาพ 2565
    - ข้อมูลทั่วไปของโรงพยาบาล 2566

    ### ยา & โรงพยาบาล
    - **บัญชียาหลักแห่งชาติ พ.ศ. 2556** (National List of Essential Medicines) — ED category ก–จ2 + หมายเหตุการเบิกจ่าย
    - `data/processed/disease_drug_mapping_v2_ed.csv` — mapping ยา–โรค (87 ยา)
    - `data/processed/hospitals_thailand.csv` — รายชื่อโรงพยาบาล 1,581 แห่งทั่วประเทศ

    ### Constructed ourselves
    - `data/processed/disease_specialty_mapping.csv` — 48 × 13
    - `data/processed/symptom_dictionary_th.csv` — 132 × 7 (เพิ่มคอลัมน์ symptom_dialect รองรับภาษาถิ่น อีสาน/เหนือ/ใต้)
    - `data/processed/symptom_specificity.csv` — 132 × 5
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
st.markdown("## Team")
st.info("👤 _เพิ่มชื่อสมาชิก + บทบาทที่นี่_")

st.markdown(
    """
    | สมาชิก | บทบาท |
    |---|---|
    | _Member 1_ | Data Engineer · ดูแล data ingestion + DuckDB |
    | _Member 2_ | Non-AI Engineer · scoring + rule engine |
    | _Member 3_ | AI Engineer · LLM pipeline + RAG |
    | _Member 4_ | UI/Visualization · Streamlit + Plotly |
    | _Member 5_ | PM/QA · slide + rehearsal |
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Tech stack
# ---------------------------------------------------------------------------
st.markdown("## Tech stack")

st.markdown(
    """
    - **Frontend / App:** Streamlit (multi-pages)
    - **Analytics:** DuckDB + pandas
    - **External DB:** Snowflake (AWS ap-southeast-1) + MongoDB Atlas (M0)
    - **AI:** Gemini Flash / Gemma (น้องอุ่นใน)
    - **Visualization:** Plotly + matplotlib + seaborn
    - **Hosting:** Streamlit Community Cloud
    - **Repo:** [github.com/gettokeroro/DADS5001_FINAL_PORJECT](https://github.com/gettokeroro/DADS5001_FINAL_PORJECT)
    """
)

st.divider()

# ---------------------------------------------------------------------------
# License
# ---------------------------------------------------------------------------
st.markdown("## License & Disclaimer")
st.info(
    "Educational use only · DADS5001 final project · "
    "ดูรายละเอียดเต็มที่ `DISCLAIMER.md` ใน repo"
)
