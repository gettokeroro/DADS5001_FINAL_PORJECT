"""
Hospital Province Filter — Demo
================================
รันด้วย: streamlit run hospital_province_demo.py

Pattern (ก) — st.selectbox built-in autocomplete
- Streamlit's selectbox มี filter built-in: พิมพ์ในช่องค้นหา → list ลดลงตามตัวอักษร
- ตั้ง index=None + placeholder เพื่อให้เริ่มต้นว่าง รอ user พิมพ์
- ใช้ session_state เก็บค่าที่เลือก (ตามโจทย์ DADS5001)
"""
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="🏥 ค้นหาโรงพยาบาลตามจังหวัด", layout="wide")
st.title("🏥 แนะนำโรงพยาบาลตามจังหวัด")
st.caption("Pattern (ก) — st.selectbox + autocomplete · พิมพ์ชื่อจังหวัดในช่องเพื่อ filter")

# ---------- Data loader (cache_data) ----------
@st.cache_data(show_spinner=False)
def load_hospitals() -> pd.DataFrame:
    """โหลด hospital CSV (UTF-8) · ใช้ cache_data ตามโจทย์"""
    candidates = [
        Path("data/processed/hospitals_thailand.csv"),
        Path(__file__).parent / "hospitals_thailand.csv",
        Path(__file__).parent / "data" / "processed" / "hospitals_thailand.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    st.error("ไม่พบไฟล์ hospitals_thailand.csv · วางที่ data/processed/")
    st.stop()


@st.cache_data(show_spinner=False)
def get_province_list(df: pd.DataFrame) -> list[str]:
    """77 จังหวัด เรียงตามตัวอักษร"""
    return sorted(df["province"].dropna().unique().tolist())


# ---------- Load ----------
df = load_hospitals()
provinces = get_province_list(df)

st.metric("ทั้งหมด", f"{len(df)} โรงพยาบาล · {len(provinces)} จังหวัด")
st.markdown("---")

# ---------- Step 1: Province selectbox (autocomplete) ----------
st.subheader("1️⃣ เลือกจังหวัดที่อยู่")

selected_province = st.selectbox(
    label="ค้นหาจังหวัด (พิมพ์เพื่อ filter)",
    options=provinces,
    index=None,
    placeholder="พิมพ์เช่น 'เชียง' จะเห็น เชียงใหม่ · เชียงราย ...",
    key="province_select",
    help="พิมพ์ตัวอักษรไทยในช่องค้นหาภายใน selectbox · จะเห็น list ที่ตรงกับสิ่งที่พิมพ์",
)

if selected_province is None:
    st.info("👆 กรุณาเลือกจังหวัดเพื่อดูรายชื่อโรงพยาบาล")
    st.stop()

# Save to session_state
st.session_state["selected_province"] = selected_province

# ---------- Step 2: Filter hospitals by province ----------
hosp_in_province = df[df["province"] == selected_province].copy()
st.success(f"พบ **{len(hosp_in_province)} โรงพยาบาล** ในจังหวัด **{selected_province}**")

# ---------- Step 3: Optional secondary filter ----------
col1, col2 = st.columns(2)

with col1:
    hosp_types = sorted(hosp_in_province["hospital_type"].dropna().unique().tolist())
    type_filter = st.selectbox(
        "🏥 กรองตามชนิด รพ. (เลือกได้)",
        options=["ทุกประเภท"] + hosp_types,
        index=0,
        key="type_select",
    )

with col2:
    affs = sorted(hosp_in_province["affiliation"].dropna().unique().tolist())
    aff_filter = st.selectbox(
        "🏛 กรองตามสังกัด (เลือกได้)",
        options=["ทุกสังกัด"] + affs,
        index=0,
        key="aff_select",
    )

# Apply secondary filters
filtered = hosp_in_province.copy()
if type_filter != "ทุกประเภท":
    filtered = filtered[filtered["hospital_type"] == type_filter]
if aff_filter != "ทุกสังกัด":
    filtered = filtered[filtered["affiliation"] == aff_filter]

st.markdown("---")

# ---------- Step 4: Display results ----------
st.subheader(f"2️⃣ รายชื่อโรงพยาบาล ({len(filtered)} แห่ง)")

if filtered.empty:
    st.warning("ไม่พบโรงพยาบาลที่ตรงเงื่อนไข · ลองปรับ filter")
else:
    # Sort by beds descending (รพ.ใหญ่ก่อน)
    filtered = filtered.sort_values("beds", ascending=False, na_position="last")

    for _, row in filtered.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{row['hospital_th']}**  \n*{row['hospital_en']}*")
                st.caption(f"📍 {row['province']} · {row['hospital_type']} · {row['affiliation']}")
                if pd.notna(row.get("specialty_note")):
                    st.markdown(f"🩺 ความเชี่ยวชาญ: {row['specialty_note']}")
            with c2:
                if pd.notna(row.get("beds")):
                    st.metric("เตียง", int(row["beds"]))
                st.caption(f"H Code: {row.get('h_code', '—')}")

# ---------- Debug expander ----------
with st.expander("🔧 รายละเอียดทางเทคนิค (debug)"):
    st.write(f"**Selected province:** `{selected_province}`")
    st.write(f"**Type filter:** `{type_filter}`")
    st.write(f"**Affiliation filter:** `{aff_filter}`")
    st.write(f"**Filtered rows:** {len(filtered)}")
    st.write("**Session state:**", dict(st.session_state))
