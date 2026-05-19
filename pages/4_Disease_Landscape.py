"""
Page 4 · Disease Landscape
แหล่งข้อมูล:
  - ha_aod_001-2.csv  (TIS-620) → top-disease mentions จาก รพ.ทั่วประเทศ
  - disease_specialty_mapping.csv → ความครอบคลุมของ app (48 โรค × 13 สาขา)
  - hospitals_thailand.csv       → hospital count per province / health region
  - MongoDB ai_sessions           → real user analytics (top diseases, symptoms, confidence)
"""
import re
import sys
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    _get_mongo_db,
    init_session_state,
    render_disclaimer_sidebar,
)

st.set_page_config(page_title="Disease Landscape", page_icon="🗺️", layout="wide")
init_session_state()
render_disclaimer_sidebar()

st.markdown("""<style>
[data-testid="stMetricDelta"] > div {font-size:10.5px !important;white-space:normal !important;overflow:visible !important;line-height:1.3 !important;}
[data-testid="stMetricValue"] {font-size:1.35rem !important;}
[data-testid="stMetricLabel"] {font-size:13px !important;white-space:normal !important;overflow:visible !important;}
div.stPlotlyChart {border:1px solid #e2e8f0;border-radius:12px;padding:8px;background:white;}
</style>""", unsafe_allow_html=True)

DATA = Path(__file__).resolve().parent.parent / "data"

NAVY  = "#0a2342"
TEAL  = "#1B9AAA"
PINK  = "#E84C8B"
AMBER = "#F7B731"
GREEN = "#26de81"


def _fig_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sarabun, sans-serif", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


DISEASE_THAI_MAP = {
    "Sepsis":         "ภาวะพิษเลือด",
    "HT":             "ความดันโลหิตสูง",
    "Stroke":         "โรคหลอดเลือดสมอง",
    "STEMI":          "กล้ามเนื้อหัวใจตายเฉียบพลัน",
    "COPD":           "ถุงลมโป่งพอง",
    "TB":             "วัณโรค",
    "DM":             "เบาหวาน",
    "CKD":            "ไตเรื้อรัง",
    "DHF":            "ไข้เลือดออก",
    "HIV":            "เอชไอวี/เอดส์",
    "PPH":            "ตกเลือดหลังคลอด",
    "Trauma":         "บาดเจ็บ/อุบัติเหตุ",
    "Pneumonia":      "ปอดบวม",
    "Asthma":         "หอบหืด",
    "PIH":            "ความดันในครรภ์",
    "Head injury":    "บาดเจ็บศีรษะ",
    "HI":             "บาดเจ็บศีรษะ",
    "ACS":            "กล้ามเนื้อหัวใจขาดเลือด",
    "AMI":            "กล้ามเนื้อหัวใจตาย",
    "Birth asphyxia": "ทารกขาดออกซิเจนแรกเกิด",
    "Appendicitis":   "ไส้ติ่งอักเสบ",
    "COVID-19":       "โควิด-19",
    "Dengue":         "ไข้เลือดออก",
}

DISEASE_TOP_SYMPTOMS = {
    "HT":        ["ปวดศีรษะ", "เวียนศีรษะ", "ใจสั่น", "เหนื่อยง่าย", "มองเห็นผิดปกติ"],
    "Stroke":    ["แขนขาอ่อนแรงซีกเดียว", "พูดไม่ชัด", "เดินเซ", "ปวดศีรษะรุนแรง", "หมดสติ"],
    "STEMI":     ["เจ็บหน้าอก", "เหนื่อยหอบ", "เหงื่อออกมาก", "คลื่นไส้", "ใจสั่น"],
    "COPD":      ["ไอเรื้อรัง", "เหนื่อยหอบ", "มีเสมหะ", "แน่นหน้าอก", "เหนื่อยง่าย"],
    "TB":        ["ไอเรื้อรัง", "ไข้ต่ำๆ", "น้ำหนักลด", "เหงื่อออกกลางคืน", "อ่อนเพลีย"],
    "DM":        ["ปัสสาวะบ่อย", "กระหายน้ำมาก", "หิวบ่อย", "น้ำหนักลด", "อ่อนเพลีย"],
    "CKD":       ["บวมขา", "ความดันสูง", "ปัสสาวะน้อย", "อ่อนเพลีย", "คลื่นไส้"],
    "DHF":       ["ไข้สูงเฉียบพลัน", "ปวดศีรษะ", "ปวดกล้ามเนื้อ", "มีผื่น", "คลื่นไส้"],
    "HIV":       ["ไข้", "อ่อนเพลีย", "ต่อมน้ำเหลืองโต", "น้ำหนักลด", "ไอ"],
    "Pneumonia": ["ไข้", "ไอ", "เหนื่อยหอบ", "เจ็บหน้าอก", "เสมหะ"],
    "Asthma":    ["หายใจมีเสียงหวีด", "หอบ", "ไอ", "แน่นหน้าอก", "เหนื่อยง่าย"],
    "Sepsis":    ["ไข้สูง/ต่ำผิดปกติ", "ใจสั่นเร็ว", "หายใจเร็ว", "ความดันต่ำ", "ซึม"],
}

PROVINCE_CENTROIDS = {
    "กรุงเทพมหานคร": (13.754, 100.501),
    "กระบี่": (8.085, 98.906),
    "กาญจนบุรี": (14.018, 99.534),
    "กาฬสินธุ์": (16.432, 103.506),
    "กำแพงเพชร": (16.483, 99.522),
    "ขอนแก่น": (16.432, 102.836),
    "จันทบุรี": (12.610, 102.104),
    "ฉะเชิงเทรา": (13.689, 101.078),
    "ชลบุรี": (13.361, 100.985),
    "ชัยนาท": (15.186, 100.125),
    "ชัยภูมิ": (15.806, 102.032),
    "ชุมพร": (10.494, 99.180),
    "เชียงราย": (19.911, 99.840),
    "เชียงใหม่": (18.787, 98.993),
    "ตรัง": (7.560, 99.624),
    "ตราด": (12.244, 102.513),
    "ตาก": (16.879, 99.126),
    "นครนายก": (14.201, 101.215),
    "นครปฐม": (13.820, 100.054),
    "นครพนม": (17.392, 104.769),
    "นครราชสีมา": (14.980, 102.098),
    "นครศรีธรรมราช": (8.432, 99.960),
    "นครสวรรค์": (15.705, 100.137),
    "นนทบุรี": (13.859, 100.525),
    "นราธิวาส": (6.426, 101.824),
    "น่าน": (18.774, 100.777),
    "บึงกาฬ": (18.361, 103.646),
    "บุรีรัมย์": (14.993, 103.102),
    "ปทุมธานี": (14.013, 100.529),
    "ประจวบคีรีขันธ์": (11.813, 99.797),
    "ปราจีนบุรี": (14.052, 101.366),
    "ปัตตานี": (6.869, 101.250),
    "พระนครศรีอยุธยา": (14.357, 100.587),
    "พะเยา": (19.163, 99.901),
    "พังงา": (8.451, 98.526),
    "พัทลุง": (7.617, 100.075),
    "พิจิตร": (16.441, 100.349),
    "พิษณุโลก": (16.829, 100.265),
    "เพชรบุรี": (13.112, 99.939),
    "เพชรบูรณ์": (16.419, 101.160),
    "แพร่": (18.145, 100.140),
    "ภูเก็ต": (7.890, 98.398),
    "มหาสารคาม": (16.185, 103.301),
    "มุกดาหาร": (16.543, 104.724),
    "แม่ฮ่องสอน": (19.301, 97.966),
    "ยโสธร": (15.792, 104.145),
    "ยะลา": (6.544, 101.281),
    "ระนอง": (9.958, 98.635),
    "ระยอง": (12.683, 101.281),
    "ราชบุรี": (13.536, 99.818),
    "ร้อยเอ็ด": (16.054, 103.652),
    "ลพบุรี": (14.799, 100.653),
    "ลำปาง": (18.292, 99.491),
    "ลำพูน": (18.574, 98.986),
    "เลย": (17.486, 101.722),
    "ศรีสะเกษ": (15.117, 104.322),
    "สกลนคร": (17.155, 104.148),
    "สงขลา": (7.190, 100.595),
    "สตูล": (6.615, 100.068),
    "สมุทรปราการ": (13.599, 100.600),
    "สมุทรสงคราม": (13.412, 100.002),
    "สมุทรสาคร": (13.547, 100.274),
    "สระแก้ว": (13.824, 102.065),
    "สระบุรี": (14.530, 100.910),
    "สิงห์บุรี": (14.891, 100.397),
    "สุโขทัย": (17.007, 99.826),
    "สุพรรณบุรี": (14.472, 100.128),
    "สุราษฎร์ธานี": (9.140, 99.330),
    "สุรินทร์": (14.882, 103.494),
    "หนองคาย": (17.877, 102.743),
    "หนองบัวลำภู": (17.204, 102.443),
    "อ่างทอง": (14.590, 100.455),
    "อำนาจเจริญ": (15.866, 104.625),
    "อุดรธานี": (17.416, 102.787),
    "อุตรดิตถ์": (17.620, 100.099),
    "อุทัยธานี": (15.380, 100.024),
    "อุบลราชธานี": (15.245, 104.847),
}


@st.cache_data(show_spinner=False)
def _load_hospital_disease_mentions():
    raw_path = DATA / "raw" / "datago" / "ha_aod_001-2.csv"
    try:
        df = pd.read_csv(raw_path, encoding="cp874", low_memory=False, header=0)
        disease_col = df.iloc[:, 19].dropna().astype(str)
        counts = Counter()
        for text in disease_col:
            parts = re.split(r"[,/\\\n;\xb7\x95]+", text)
            for p in parts:
                p = p.strip().strip('"').strip("'").strip()
                if (len(p) >= 2
                        and p.lower() not in ("nan", "", "-", "null", "none")
                        and "<" not in p and ">" not in p
                        and "br" not in p.lower()):
                    counts[p] += 1
        if not counts:
            return pd.DataFrame(columns=["disease", "disease_th", "count", "label"])
        result = pd.DataFrame(counts.most_common(25), columns=["disease", "count"])
        result["disease_th"] = result["disease"].map(DISEASE_THAI_MAP)
        result["label"] = result.apply(
            lambda r: f"{r['disease_th']}\n({r['disease']})" if pd.notna(r["disease_th"]) else r["disease"],
            axis=1,
        )
        return result
    except Exception:
        return pd.DataFrame(columns=["disease", "disease_th", "count", "label"])


@st.cache_data(show_spinner=False)
def _load_disease_specialty_pivot():
    path = DATA / "processed" / "disease_specialty_mapping.csv"
    try:
        df = pd.read_csv(path)
        all_specs = sorted(set(df["primary_specialty"].dropna().unique()))
        def _short(s):
            return re.sub(r"\s*\(.*?\)", "", s).strip()
        short_specs = {s: _short(s) for s in all_specs}
        cols = [short_specs[s] for s in all_specs]
        rows = []
        for _, row in df.iterrows():
            vec = {c: 0 for c in cols}
            ps = short_specs.get(row["primary_specialty"], "")
            ss_raw = str(row.get("secondary_specialty", ""))
            ss = short_specs.get(ss_raw, "")
            if ps in vec:
                vec[ps] = 2
            if ss and ss in vec:
                vec[ss] = max(vec[ss], 1)
            rows.append({"disease": row["disease_th"], **vec})
        pivot = pd.DataFrame(rows).set_index("disease")
        return pivot
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_hospital_province():
    path = DATA / "processed" / "hospitals_thailand.csv"
    try:
        df = pd.read_csv(path)
        return (
            df.groupby(["health_region", "province"])
            .size()
            .reset_index(name="n_hospitals")
            .sort_values(["health_region", "n_hospitals"], ascending=[True, False])
        )
    except Exception:
        return pd.DataFrame(columns=["health_region", "province", "n_hospitals"])


@st.cache_data(show_spinner=False)
def _load_hospital_raw():
    path = DATA / "processed" / "hospitals_thailand.csv"
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def _load_thailand_geojson():
    """Fetch Thailand province GeoJSON (cached 24 h). Runs on Streamlit Cloud."""
    import urllib.request as _urlreq, json as _json
    url = "https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json"
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _load_ai_sessions_analytics():
    try:
        db = _get_mongo_db()
        if db is None:
            return {"status": "unavailable"}
        docs = list(db["ai_sessions"].find(
            {},
            {"_id": 0, "timestamp": 1, "symptoms": 1, "top_disease": 1,
             "confidence_level": 1, "n_symptoms": 1}
        ))
        if not docs:
            return {"status": "empty"}
        df = pd.DataFrame(docs)
        disease_counts = (
            df["top_disease"].value_counts().head(15)
            .reset_index().rename(columns={"top_disease": "disease", "count": "n"})
        )
        all_sym = []
        for row in df["symptoms"].dropna():
            if isinstance(row, list):
                all_sym.extend(row)
        symptom_df = (
            pd.DataFrame(Counter(all_sym).most_common(15), columns=["symptom", "n"])
            if all_sym else pd.DataFrame(columns=["symptom", "n"])
        )
        conf_counts = (
            df["confidence_level"].value_counts()
            .reset_index().rename(columns={"confidence_level": "level", "count": "n"})
        )
        lbl = {"high": "มั่นใจสูง", "medium": "มั่นใจปานกลาง", "low": "มั่นใจต่ำ", "very_low": "มั่นใจต่ำมาก"}
        conf_counts["label"] = conf_counts["level"].map(lbl).fillna(conf_counts["level"])
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date").size().reset_index(name="sessions")
        daily["date"] = pd.to_datetime(daily["date"])
        return {
            "status": "ok", "total": len(df),
            "top_diseases": disease_counts, "top_symptoms": symptom_df,
            "confidence_dist": conf_counts, "daily_trend": daily,
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}



# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("\U0001f5fa\ufe0f Disease Landscape Thailand")
st.markdown("##### \u0e20\u0e32\u0e1e\u0e23\u0e27\u0e21\u0e42\u0e23\u0e04\u0e43\u0e19\u0e1b\u0e23\u0e30\u0e40\u0e17\u0e28\u0e44\u0e17\u0e22 \xb7 \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e08\u0e32\u0e01\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e08\u0e23\u0e34\u0e07 + MongoDB User Analytics")
st.divider()

with st.expander("\U0001f4cc \u0e2a\u0e16\u0e34\u0e15\u0e34\u0e42\u0e23\u0e04\u0e44\u0e17\u0e22 2024\u20132025 (\u0e2d\u0e49\u0e32\u0e07\u0e2d\u0e34\u0e07\u0e08\u0e32\u0e01 DDC \xb7 WHO \xb7 IDF)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001f480 \u0e40\u0e2a\u0e35\u0e22\u0e0a\u0e35\u0e27\u0e34\u0e15\u0e08\u0e32\u0e01 NCDs/\u0e1b\u0e35", "~420,000 \u0e04\u0e19", "74% \u0e02\u0e2d\u0e07\u0e01\u0e32\u0e23\u0e40\u0e2a\u0e35\u0e22\u0e0a\u0e35\u0e27\u0e34\u0e15\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14")
    c2.metric("\U0001f9b8 \u0e1c\u0e39\u0e49\u0e1b\u0e48\u0e27\u0e22\u0e40\u0e1a\u0e32\u0e2b\u0e27\u0e32\u0e19", "6.36 \u0e25\u0e49\u0e32\u0e19\u0e04\u0e19", "\u0e2d\u0e31\u0e19\u0e14\u0e31\u0e1a 4 Western Pacific \xb7 \u0e1b\u0e35 2024")
    c3.metric("\u2764\ufe0f \u0e1c\u0e39\u0e49\u0e1b\u0e48\u0e27\u0e22\u0e04\u0e27\u0e32\u0e21\u0e14\u0e31\u0e19\u0e42\u0e25\u0e2b\u0e34\u0e15\u0e2a\u0e39\u0e07", "~13.7 \u0e25\u0e49\u0e32\u0e19\u0e04\u0e19", "\u0e40\u0e1e\u0e34\u0e48\u0e21\u0e02\u0e36\u0e49\u0e19\u0e15\u0e48\u0e2d\u0e40\u0e19\u0e37\u0e48\u0e2d\u0e07\u0e17\u0e38\u0e01\u0e1b\u0e35")
    c4.metric("\U0001f3e5 \u0e20\u0e32\u0e23\u0e30\u0e42\u0e23\u0e04 NCDs \u0e23\u0e27\u0e21", "~14 \u0e25\u0e49\u0e32\u0e19\u0e04\u0e19", "\u0e40\u0e1a\u0e32\u0e2b\u0e27\u0e32\u0e19 \xb7 \u0e04\u0e27\u0e32\u0e21\u0e14\u0e31\u0e19 \xb7 \u0e2b\u0e31\u0e27\u0e43\u0e08 \xb7 \u0e44\u0e15")
    st.caption("Source: \u0e01\u0e23\u0e21\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21\u0e42\u0e23\u0e04 (DDC) \xb7 WHO Thailand 2025 \xb7 IDF Diabetes Atlas 2024 \xb7 \u0e2a\u0e2a\u0e2a. 2568")
    st.info("\u2139\ufe0f \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25 2 \u0e0a\u0e38\u0e14\u0e02\u0e49\u0e32\u0e07\u0e15\u0e49\u0e19\u0e21\u0e32\u0e08\u0e32\u0e01\u0e04\u0e19\u0e25\u0e30\u0e41\u0e2b\u0e25\u0e48\u0e07: \u0e15\u0e31\u0e27\u0e40\u0e25\u0e02 DDC/WHO = \u0e2a\u0e16\u0e34\u0e15\u0e34\u0e23\u0e30\u0e14\u0e31\u0e1a\u0e1b\u0e23\u0e30\u0e40\u0e17\u0e28 (\u0e20\u0e32\u0e23\u0e30\u0e42\u0e23\u0e04 NCD) \xb7 \u0e01\u0e23\u0e32\u0e1f\u0e14\u0e49\u0e32\u0e19\u0e25\u0e48\u0e32\u0e07 = \u0e01\u0e25\u0e38\u0e48\u0e21\u0e1c\u0e39\u0e49\u0e1b\u0e48\u0e27\u0e22\u0e17\u0e35\u0e48 \u0e23\u0e1e. \u0e23\u0e32\u0e22\u0e07\u0e32\u0e19\u0e40\u0e2d\u0e07 (ha_aod \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e08\u0e23\u0e34\u0e07)")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f3e5 \u0e42\u0e23\u0e04\u0e43\u0e19 \u0e23\u0e1e.\u0e44\u0e17\u0e22",
    "\U0001f4ca \u0e20\u0e32\u0e1e\u0e23\u0e27\u0e21\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25",
    "\U0001f4cd \u0e41\u0e1c\u0e19\u0e17\u0e35\u0e48\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25",
    "\U0001f4ca User Analytics",
])

# ==========================================================================
# Tab 1
# ==========================================================================
with tab1:
    st.markdown("### \U0001f3e5 \u0e42\u0e23\u0e04\u0e17\u0e35\u0e48\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e44\u0e17\u0e22\u0e23\u0e32\u0e22\u0e07\u0e32\u0e19\u0e27\u0e48\u0e32\u0e1e\u0e1a\u0e1a\u0e48\u0e2d\u0e22")
    st.caption("Source: \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e2a\u0e31\u0e07\u0e01\u0e31\u0e14 \u0e2a\u0e18. (ha_aod_001-2) \xb7 \u0e01\u0e25\u0e38\u0e48\u0e21\u0e1c\u0e39\u0e49\u0e1b\u0e48\u0e27\u0e22\u0e42\u0e23\u0e04\u0e2a\u0e33\u0e04\u0e31\u0e0d \u2014 437 \u0e23\u0e1e. \u0e17\u0e31\u0e48\u0e27\u0e1b\u0e23\u0e30\u0e40\u0e17\u0e28")

    with st.spinner("\u0e01\u0e33\u0e25\u0e31\u0e07\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25..."):
        mentions_df = _load_hospital_disease_mentions()

    if mentions_df.empty:
        st.warning("\u0e44\u0e21\u0e48\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25 \u2014 \u0e15\u0e23\u0e27\u0e08\u0e2a\u0e2d\u0e1a\u0e44\u0e1f\u0e25\u0e4c ha_aod_001-2.csv")
    else:
        topn = st.slider("\u0e41\u0e2a\u0e14\u0e07 Top N \u0e42\u0e23\u0e04", 5, 25, 10, key="tab1_topn")
        chart_df = mentions_df.head(topn).sort_values("count", ascending=False).copy()
        chart_df["label_short"] = chart_df["disease_th"].fillna(chart_df["disease"])
        fig = px.bar(
            chart_df,
            x="label_short", y="count",
            color="count",
            color_continuous_scale=[[0, "#c7ecf7"], [1, TEAL]],
            text="count",
            title=f"Top {topn} โรคที่ รพ.ไทยรายงานในกลุ่มผู้ป่วยสำคัญ",
            labels={"count": "จำนวนโรงพยาบาลที่รายงาน", "label_short": ""},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            xaxis=dict(tickfont=dict(size=10), title="", tickangle=-35),
            yaxis=dict(title="จำนวน รพ."),
            height=520,
        )
        _fig_style(fig)
        fig.update_layout(margin=dict(t=60, b=110))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**\u0e2a\u0e23\u0e38\u0e1b Top \u0e42\u0e23\u0e04\u0e41\u0e25\u0e30\u0e2d\u0e32\u0e01\u0e32\u0e23\u0e2a\u0e33\u0e04\u0e31\u0e0d**")
        summary_rows = []
        for _, r in mentions_df.head(topn).iterrows():
            th = DISEASE_THAI_MAP.get(r["disease"], r["disease"])
            syms = DISEASE_TOP_SYMPTOMS.get(r["disease"])
            sym_str = " \xb7 ".join(syms[:3]) if syms else "\u2014"
            summary_rows.append({
                "\u0e42\u0e23\u0e04 (\u0e44\u0e17\u0e22)": th,
                "\u0e23\u0e1e.\u0e17\u0e35\u0e48\u0e23\u0e32\u0e22\u0e07\u0e32\u0e19": f"{int(r['count'])}",
                "\u0e2d\u0e32\u0e01\u0e32\u0e23\u0e2b\u0e25\u0e31\u0e01": sym_str,
            })
        _, col_tbl, _ = st.columns([1, 3, 1])
        with col_tbl:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.info("\U0001f4a1 \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e08\u0e32\u0e01\u0e1f\u0e34\u0e25\u0e14\u0e4c '\u0e01\u0e25\u0e38\u0e48\u0e21\u0e1c\u0e39\u0e49\u0e1b\u0e48\u0e27\u0e22\u0e42\u0e23\u0e04\u0e2a\u0e33\u0e04\u0e31\u0e0d' \u0e17\u0e35\u0e48\u0e41\u0e15\u0e48\u0e25\u0e30\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e23\u0e32\u0e22\u0e07\u0e32\u0e19\u0e40\u0e2d\u0e07 \u2014 \u0e2a\u0e30\u0e17\u0e49\u0e2d\u0e19\u0e42\u0e23\u0e04\u0e17\u0e35\u0e48 \u0e23\u0e1e. \u0e40\u0e2b\u0e47\u0e19\u0e08\u0e23\u0e34\u0e07\u0e43\u0e19\u0e17\u0e32\u0e07\u0e1b\u0e0f\u0e34\u0e1a\u0e31\u0e15\u0e34")


# ==========================================================================
# Tab 2 — ภาพรวมข้อมูล
# ==========================================================================
with tab2:
    # ── KPI row ────────────────────────────────────────────────────────────
    _k1, _k2, _k3, _k4 = st.columns(4)
    with _k1:
        st.metric("🦠 โรค", "48 โรค", delta="ครอบคลุมทุกกลุ่ม ICD-10 หลัก", delta_color="off")
    with _k2:
        st.metric("🩺 อาการ", "131 อาการ", delta="42 โรค · รวมอาการซ้ำข้ามโรค", delta_color="off")
    with _k3:
        st.metric("🏥 โรงพยาบาล", "1,581 รพ.", delta="ข้อมูลจริงทั่วประเทศ", delta_color="off")
    with _k4:
        st.metric("📍 จังหวัด", "77 จังหวัด", delta="13 เขตสุขภาพ", delta_color="off")

    st.divider()

    # ── Row A: Treemap ICD-10  |  อาการที่พบบ่อย ──────────────────────────
    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("#### 48 โรคตามกลุ่ม ICD-10")
        st.caption("disease_specialty_mapping.csv · แต่ละสี = กลุ่ม ICD-10 · คลิกเพื่อ zoom in/out")
        try:
            _ds = pd.read_csv(DATA / "processed" / "disease_specialty_mapping.csv")
            _ds["icd10_chapter_name"] = _ds["icd10_chapter_name"].str.replace(
                r"โรคระบบกล้ามเนื้อ กระดูก.*", "โรคระบบกล้ามเนื้อ กระดูก", regex=True
            )
            _ds["_val"] = 1
            fig_tree = px.treemap(
                _ds,
                path=["icd10_chapter_name", "disease_th"],
                values="_val",
                color="icd10_chapter_name",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_tree.update_traces(
                textinfo="label",
                marker=dict(line=dict(width=2, color="white")),
                hovertemplate="<b>%{label}</b><br>กลุ่ม: %{parent}<extra></extra>",
            )
            fig_tree.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            _fig_style(fig_tree)
            st.plotly_chart(fig_tree, use_container_width=True)
        except Exception as _e:
            st.warning(f"โหลด disease specialty data ไม่ได้: {_e}")

    with col_a2:
        st.markdown("#### อาการที่ปรากฏในหลายโรค")
        st.caption("จาก disease_symptom_long.csv · นับจำนวนโรคที่มีอาการนี้")
        _SYM_TH = {
            "fatigue": "อ่อนเพลีย", "vomiting": "อาเจียน",
            "high_fever": "ไข้สูง", "headache": "ปวดศีรษะ",
            "nausea": "คลื่นไส้", "loss_of_appetite": "เบื่ออาหาร",
            "abdominal_pain": "ปวดท้อง", "chills": "หนาวสั่น",
            "yellowish_skin": "ผิวเหลือง", "skin_rash": "ผื่นผิวหนัง",
            "fever": "ไข้", "cough": "ไอ",
            "breathlessness": "หายใจลำบาก", "chest_pain": "เจ็บหน้าอก",
            "weight_loss": "น้ำหนักลด", "sweating": "เหงื่อออก",
            "muscle_pain": "ปวดกล้ามเนื้อ", "diarrhoea": "ท้องเสีย",
            "joint_pain": "ปวดข้อ", "swelling": "บวม",
        }
        try:
            _sym_df = pd.read_csv(DATA / "processed" / "disease_symptom_long.csv")
            _sym_cnt = (
                _sym_df.groupby("symptom")["disease"].nunique()
                .reset_index(name="n_diseases")
                .sort_values("n_diseases", ascending=False)
                .head(12)
            )
            _sym_cnt["label"] = _sym_cnt["symptom"].map(_SYM_TH).fillna(_sym_cnt["symptom"])
            fig_sym = px.bar(
                _sym_cnt.sort_values("n_diseases"),
                x="n_diseases", y="label", orientation="h",
                color="n_diseases",
                color_continuous_scale=[[0, "#c7ecf7"], [1, PINK]],
                text="n_diseases",
                labels={"n_diseases": "จำนวนโรค", "label": ""},
            )
            fig_sym.update_traces(texttemplate="%{text} โรค", textposition="outside")
            fig_sym.update_coloraxes(showscale=False)
            fig_sym.update_layout(
                height=520,
                yaxis=dict(tickfont=dict(size=11)),
                xaxis=dict(title="จำนวนโรคที่มีอาการนี้"),
                margin=dict(l=10, r=80, t=30, b=30),
            )
            _fig_style(fig_sym)
            st.plotly_chart(fig_sym, use_container_width=True)
        except Exception as _e:
            st.warning(f"โหลด symptom data ไม่ได้: {_e}")

    st.divider()

    # ── Row B: Treemap ICD→โรค→อาการ  |  Donut สาขาแพทย์ ──────────────
    col_b1, col_b2 = st.columns([3, 2])

    with col_b1:
        st.markdown("#### ICD-10 → โรค → อาการหลัก")
        st.caption("คลิกเพื่อ zoom · ขนาด = ความถี่อาการในโรคนั้น")
        try:
            import re as _re
            _ds3 = pd.read_csv(DATA / "processed" / "disease_specialty_mapping.csv")
            _ds3["icd10_chapter_name"] = _ds3["icd10_chapter_name"].str.replace(
                r"โรคระบบกล้ามเนื้อ กระดูก.*", "โรคระบบกล้ามเนื้อ กระดูก", regex=True
            )
            _sym3 = pd.read_csv(DATA / "processed" / "disease_symptom_long.csv")
            _BRIDGE = {
                "(vertigo) Paroymsal  Positional Vertigo": "Benign Paroxysmal Positional Vertigo (BPPV)",
                "AIDS": "HIV/AIDS", "Arthritis": "Arthritis (general)",
                "Chicken pox": "Chicken pox (Varicella)", "Dengue": "Dengue fever",
                "Diabetes ": "Diabetes mellitus",
                "Dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
                "Heart attack": "Acute myocardial infarction",
                "Hypertension ": "Hypertension", "Malaria": "Malaria",
                "Osteoarthristis": "Osteoarthritis",
                "Paralysis (brain hemorrhage)": "Paralysis (brain hemorrhage)",
                "Peptic ulcer diseae": "Peptic ulcer disease",
                "Typhoid": "Typhoid fever",
                "Urinary tract infection": "Urinary tract infection (UTI)",
                "hepatitis A": "Hepatitis A",
            }
            _SYM_TH2 = {
                "fatigue": "อ่อนเพลีย", "lethargy": "อ่อนเพลีย", "malaise": "อ่อนเพลีย",
                "high_fever": "ไข้สูง", "mild_fever": "ไข้ต่ำๆ",
                "chills": "หนาวสั่น", "shivering": "หนาวสั่น",
                "abdominal_pain": "ปวดท้อง", "stomach_pain": "ปวดท้อง", "belly_pain": "ปวดท้อง",
                "nausea": "คลื่นไส้", "vomiting": "อาเจียน",
                "phlegm": "มีเสมหะ", "mucoid_sputum": "มีเสมหะ", "rusty_sputum": "มีเสมหะ",
                "muscle_weakness": "กล้ามเนื้ออ่อนแรง", "muscle_wasting": "กล้ามเนื้ออ่อนแรง",
                "weakness_in_limbs": "แขนขาอ่อนแรง",
                "swollen_extremeties": "บวมแขนขา", "swollen_legs": "บวมขา",
                "swelling_joints": "ข้อบวม",
                "skin_rash": "ผื่นผิวหนัง",
                "nodal_skin_eruptions": "ผื่นตุ่มนูน",
                "red_spots_over_body": "จุดแดงตามตัว",
                "yellowish_skin": "ผิวเหลือง",
                "itching": "คัน", "skin_peeling": "ผิวลอก",
                "blackheads": "สิวหัวดำ", "pus_filled_pimples": "สิวหัวหนอง",
                "silver_like_dusting": "ผิวเป็นฝ้า", "blister": "ตุ่มน้ำพอง",
                "dischromic _patches": "ผิวด่าง", "scurring": "สะเก็ด",
                "yellow_crust_ooze": "สะเก็ดเหลือง", "red_sore_around_nose": "แผลแดงรอบจมูก",
                "yellowing_of_eyes": "ตาเหลือง",
                "blurred_and_distorted_vision": "ตามัว",
                "visual_disturbances": "มองเห็นผิดปกติ",
                "redness_of_eyes": "ตาแดง", "watering_from_eyes": "น้ำตาไหล",
                "pain_behind_the_eyes": "ปวดหลังตา",
                "dizziness": "เวียนศีรษะ", "spinning_movements": "บ้านหมุน",
                "loss_of_balance": "เสียการทรงตัว", "unsteadiness": "เดินเซ",
                "loss_of_concentration": "ขาดสมาธิ", "lack_of_concentration": "ขาดสมาธิ",
                "slurred_speech": "พูดไม่ชัด",
                "weakness_of_one_body_side": "อ่อนแรงซีกเดียว",
                "altered_sensorium": "ความรู้สึกตัวเปลี่ยน", "coma": "หมดสติ",
                "stiff_neck": "คอแข็ง", "movement_stiffness": "ข้อตึง",
                "loss_of_smell": "ได้กลิ่นลดลง",
                "cough": "ไอ", "breathlessness": "หายใจลำบาก",
                "congestion": "คัดจมูก", "runny_nose": "น้ำมูกไหล",
                "continuous_sneezing": "จาม", "sinus_pressure": "ปวดไซนัส",
                "throat_irritation": "คอระคาย", "patches_in_throat": "ตุ่มในลำคอ",
                "blood_in_sputum": "ไอเป็นเลือด",
                "indigestion": "อาหารไม่ย่อย", "acidity": "กรดไหลย้อน",
                "constipation": "ท้องผูก", "diarrhoea": "ท้องเสีย",
                "passage_of_gases": "ท้องอืด", "swelling_of_stomach": "ท้องบวม",
                "distention_of_abdomen": "ท้องโป่ง",
                "stomach_bleeding": "เลือดออกในกระเพาะ",
                "bloody_stool": "ถ่ายเป็นเลือด",
                "pain_during_bowel_movements": "ปวดขณะถ่าย",
                "pain_in_anal_region": "ปวดทวาร", "irritation_in_anus": "คันทวาร",
                "internal_itching": "คันภายใน", "acute_liver_failure": "ตับวาย",
                "chest_pain": "เจ็บหน้าอก", "fast_heart_rate": "หัวใจเต้นเร็ว",
                "palpitations": "ใจสั่น", "swollen_blood_vessels": "หลอดเลือดโป่ง",
                "prominent_veins_on_calf": "เส้นเลือดขาโป่ง",
                "cold_hands_and_feets": "มือเท้าเย็น", "bruising": "รอยช้ำ",
                "weight_loss": "น้ำหนักลด", "weight_gain": "น้ำหนักเพิ่ม",
                "obesity": "อ้วน", "excessive_hunger": "หิวมาก",
                "increased_appetite": "อยากอาหารมาก", "loss_of_appetite": "เบื่ออาหาร",
                "dehydration": "ขาดน้ำ", "enlarged_thyroid": "ต่อมไทรอยด์โต",
                "irregular_sugar_level": "น้ำตาลผิดปกติ",
                "fluid_overload.1": "บวมน้ำ", "puffy_face_and_eyes": "หน้าตาบวม",
                "sunken_eyes": "ตาโหล",
                "headache": "ปวดศีรษะ", "back_pain": "ปวดหลัง",
                "neck_pain": "ปวดคอ", "joint_pain": "ปวดข้อ",
                "muscle_pain": "ปวดกล้ามเนื้อ", "knee_pain": "ปวดเข่า",
                "hip_joint_pain": "ปวดสะโพก", "cramps": "เป็นตะคริว",
                "painful_walking": "เดินเจ็บ",
                "swelled_lymph_nodes": "ต่อมน้ำเหลืองโต",
                "anxiety": "วิตกกังวล", "depression": "ซึมเศร้า",
                "irritability": "หงุดหงิด", "mood_swings": "อารมณ์แปรปรวน",
                "restlessness": "กระสับกระส่าย",
                "brittle_nails": "เล็บเปราะ", "inflammatory_nails": "เล็บอักเสบ",
                "small_dents_in_nails": "เล็บเป็นหลุม",
                "sweating": "เหงื่อออก", "drying_and_tingling_lips": "ปากแห้งชา",
                "ulcers_on_tongue": "แผลในปาก",
                "toxic_look_(typhos)": "หน้าตาซีดหมอง",
                "history_of_alcohol_consumption": "ดื่มแอลกอฮอล์",
                "extra_marital_contacts": "มีคู่นอนหลายคน",
                "receiving_blood_transfusion": "รับเลือด",
                "receiving_unsterile_injections": "ฉีดยาไม่ปลอดเชื้อ",
                "family_history": "ประวัติครอบครัว",
                "polyuria": "ปัสสาวะบ่อย",
                "burning_micturition": "แสบเวลาปัสสาวะ",
                "continuous_feel_of_urine": "ปวดปัสสาวะตลอดเวลา",
                "bladder_discomfort": "ไม่สบายกระเพาะปัสสาวะ",
                "dark_urine": "ปัสสาวะสีเข้ม",
                "yellow_urine": "ปัสสาวะสีเหลือง",
                "foul_smell_of urine": "ปัสสาวะมีกลิ่น",
                "spotting_ urination": "ปัสสาวะกะปริด",
                "abnormal_menstruation": "ประจำเดือนผิดปกติ",
            }
            # Bridge + map + merge
            _sym3["disease_en_clean"] = _sym3["disease"].map(_BRIDGE).fillna(_sym3["disease"].str.strip())
            _sym3["symptom_th"] = _sym3["symptom"].map(_SYM_TH2).fillna(_sym3["symptom"])
            # Deduplicate: same disease + same Thai label → keep max freq
            _sym3 = (
                _sym3.groupby(["disease_en_clean", "symptom_th"])["freq"]
                .max().reset_index()
            )
            # Keep top 6 symptoms per disease by freq
            _sym3 = (
                _sym3.sort_values("freq", ascending=False)
                .groupby("disease_en_clean").head(6)
                .reset_index(drop=True)
            )
            # Merge with ICD chapter
            _sym3 = _sym3.merge(
                _ds3[["disease_en_clean", "disease_th", "icd10_chapter_name"]],
                on="disease_en_clean", how="left"
            )
            _sym3 = _sym3.dropna(subset=["disease_th"])
            fig_tree2 = px.treemap(
                _sym3,
                path=["icd10_chapter_name", "disease_th", "symptom_th"],
                values="freq",
                color="icd10_chapter_name",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_tree2.update_traces(
                textinfo="label",
                marker=dict(line=dict(width=1, color="white")),
                hovertemplate="<b>%{label}</b><br>%{parent}<br>freq: %{value:.2f}<extra></extra>",
            )
            fig_tree2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            _fig_style(fig_tree2)
            st.plotly_chart(fig_tree2, use_container_width=True)
        except Exception as _e:
            st.warning(f"โหลด treemap อาการไม่ได้: {_e}")

    with col_b2:
        st.markdown("#### การกระจายโรคตามสาขาแพทย์")
        st.caption("primary specialty · 48 โรค · 18 สาขา")
        _SPEC_TH = {
            "Hepatology": "ตับ/GI",
            "GP": "แพทย์ทั่วไป",
            "ER": "ฉุกเฉิน",
            "Dermatology": "ผิวหนัง",
            "Pulmonology": "ปอด",
            "Endocrinology": "ต่อมไร้ท่อ",
            "ID": "โรคติดเชื้อ",
            "ENT": "หู คอ จมูก",
            "Internal Medicine": "อายุรกรรม",
            "GI": "ทางเดินอาหาร",
            "Orthopedics": "กระดูก/ข้อ",
            "Allergy/Immunology": "ภูมิแพ้",
            "Cardiology": "หัวใจ",
            "Neurology": "ระบบประสาท",
            "General Surgery": "ศัลยกรรม",
            "Vascular Surgery": "หลอดเลือด",
            "Rheumatology": "รูมาโต",
            "Ophthalmology": "ตา",
        }
        try:
            _ds4 = pd.read_csv(DATA / "processed" / "disease_specialty_mapping.csv")
            def _short_spec2(s):
                import re as _re
                m = _re.search(r"\(([^)]+)\)", str(s))
                return m.group(1) if m else str(s).split()[0]
            _spec_cnt2 = (
                _ds4.groupby("primary_specialty")["disease_th"].count()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
            )
            _spec_cnt2["label_en"] = _spec_cnt2["primary_specialty"].apply(_short_spec2)
            _spec_cnt2["label"] = _spec_cnt2["label_en"].map(_SPEC_TH).fillna(_spec_cnt2["label_en"])
            fig_donut2 = px.pie(
                _spec_cnt2,
                names="label", values="n",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_donut2.update_traces(
                textposition="outside",
                textinfo="label+value",
                hovertemplate="<b>%{label}</b><br>%{value} โรค (%{percent})<extra></extra>",
            )
            fig_donut2.update_layout(
                height=520,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            _fig_style(fig_donut2)
            st.plotly_chart(fig_donut2, use_container_width=True)
        except Exception as _e:
            st.warning(f"โหลด specialty data ไม่ได้: {_e}")


# Tab 3 — Map
# ==========================================================================
with tab3:
    st.markdown("### \U0001f4cd \u0e01\u0e32\u0e23\u0e01\u0e23\u0e30\u0e08\u0e32\u0e22\u0e15\u0e31\u0e27\u0e02\u0e2d\u0e07\u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e17\u0e31\u0e48\u0e27\u0e1b\u0e23\u0e30\u0e40\u0e17\u0e28")
    st.caption("Source: hospitals_thailand.csv \xb7 1,581 \u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25 \xb7 77 \u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14 \xb7 13 \u0e40\u0e02\u0e15\u0e2a\u0e38\u0e02\u0e20\u0e32\u0e1e")

    hosp_df  = _load_hospital_province()
    hosp_raw = _load_hospital_raw()

    if hosp_df.empty:
        st.warning("\u0e44\u0e21\u0e48\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25 hospitals_thailand.csv")
    else:
        hosp_df["lat"] = hosp_df["province"].map(lambda p: PROVINCE_CENTROIDS.get(p, (None, None))[0])
        hosp_df["lon"] = hosp_df["province"].map(lambda p: PROVINCE_CENTROIDS.get(p, (None, None))[1])
        map_df = hosp_df.dropna(subset=["lat", "lon"]).copy()

        # Precompute hospital points with jitter (shared by both map paths)
        if not hosp_raw.empty:
            hosp_pts = hosp_raw.copy()
            hosp_pts["lat"] = hosp_pts["province"].map(lambda p: PROVINCE_CENTROIDS.get(p, (None, None))[0])
            hosp_pts["lon"] = hosp_pts["province"].map(lambda p: PROVINCE_CENTROIDS.get(p, (None, None))[1])
            hosp_pts = hosp_pts.dropna(subset=["lat", "lon"]).copy()
            import hashlib as _hl
            def _j(v, i, s=0.18):
                h = int(_hl.md5(f"{v}{i}".encode()).hexdigest(), 16)
                return (h % 1000 / 1000 - 0.5) * s
            hosp_pts["lat_j"] = [hosp_pts.iloc[i]["lat"] + _j(hosp_pts.iloc[i]["province"], i) for i in range(len(hosp_pts))]
            hosp_pts["lon_j"] = [hosp_pts.iloc[i]["lon"] + _j(hosp_pts.iloc[i]["province"], i + 9999) for i in range(len(hosp_pts))]
            hosp_pts["beds_str"] = hosp_pts["beds"].apply(lambda b: f"{int(b):,} เตียง" if pd.notna(b) and b > 0 else "—")
            hosp_pts["hover"] = hosp_pts.apply(
                lambda r: f"<b>{r['hospital_th']}</b><br>ประเภท: {r['hospital_type']}<br>เตียง: {r['beds_str']}<br>สังกัด: {str(r['affiliation'])[:30]}",
                axis=1,
            )
        else:
            hosp_pts = None

        # Province layer — auto-detect GeoJSON name format, use numeric IDs for robust matching
        geojson = _load_thailand_geojson()
        _PROV_TH_EN = {
            "กระบี่": "Krabi", "กรุงเทพมหานคร": "Bangkok", "กาญจนบุรี": "Kanchanaburi",
            "กาฬสินธุ์": "Kalasin", "กำแพงเพชร": "Kamphaeng Phet", "ขอนแก่น": "Khon Kaen",
            "จันทบุรี": "Chanthaburi", "ฉะเชิงเทรา": "Chachoengsao", "ชลบุรี": "Chon Buri",
            "ชัยนาท": "Chai Nat", "ชัยภูมิ": "Chaiyaphum", "ชุมพร": "Chumphon",
            "ตรัง": "Trang", "ตราด": "Trat", "ตาก": "Tak",
            "นครนายก": "Nakhon Nayok", "นครปฐม": "Nakhon Pathom", "นครพนม": "Nakhon Phanom",
            "นครราชสีมา": "Nakhon Ratchasima", "นครศรีธรรมราช": "Nakhon Si Thammarat",
            "นครสวรรค์": "Nakhon Sawan", "นนทบุรี": "Nonthaburi", "นราธิวาส": "Narathiwat",
            "น่าน": "Nan", "บึงกาฬ": "Bueng Kan", "บุรีรัมย์": "Buri Ram",
            "ปทุมธานี": "Pathum Thani", "ประจวบคีรีขันธ์": "Prachuap Khiri Khan",
            "ปราจีนบุรี": "Prachin Buri", "ปัตตานี": "Pattani",
            "พระนครศรีอยุธยา": "Phra Nakhon Si Ayutthaya", "พะเยา": "Phayao",
            "พังงา": "Phang Nga", "พัทลุง": "Phatthalung", "พิจิตร": "Phichit",
            "พิษณุโลก": "Phitsanulok", "ภูเก็ต": "Phuket",
            "มหาสารคาม": "Maha Sarakham", "มุกดาหาร": "Mukdahan", "ยะลา": "Yala",
            "ยโสธร": "Yasothon", "ระนอง": "Ranong", "ระยอง": "Rayong",
            "ราชบุรี": "Ratchaburi", "ร้อยเอ็ด": "Roi Et", "ลพบุรี": "Lop Buri",
            "ลำปาง": "Lampang", "ลำพูน": "Lamphun", "ศรีสะเกษ": "Si Sa Ket",
            "สกลนคร": "Sakon Nakhon", "สงขลา": "Songkhla", "สตูล": "Satun",
            "สมุทรปราการ": "Samut Prakan", "สมุทรสงคราม": "Samut Songkhram",
            "สมุทรสาคร": "Samut Sakhon", "สระบุรี": "Saraburi", "สระแก้ว": "Sa Kaeo",
            "สิงห์บุรี": "Sing Buri", "สุพรรณบุรี": "Suphan Buri",
            "สุราษฎร์ธานี": "Surat Thani", "สุรินทร์": "Surin", "สุโขทัย": "Sukhothai",
            "หนองคาย": "Nong Khai", "หนองบัวลำภู": "Nong Bua Lam Phu",
            "อำนาจเจริญ": "Amnat Charoen", "อุดรธานี": "Udon Thani",
            "อุตรดิตถ์": "Uttaradit", "อุทัยธานี": "Uthai Thani",
            "อุบลราชธานี": "Ubon Ratchathani", "อ่างทอง": "Ang Thong",
            "เชียงราย": "Chiang Rai", "เชียงใหม่": "Chiang Mai",
            "เพชรบุรี": "Phetchaburi", "เพชรบูรณ์": "Phetchabun",
            "เลย": "Loei", "แพร่": "Phrae", "แม่ฮ่องสอน": "Mae Hong Son",
        }
        if geojson is not None:
            # Stamp numeric IDs onto features so we can match by int (no string encoding issues)
            for _idx, _ft in enumerate(geojson["features"]):
                _ft["id"] = _idx
            _geo_names = [_ft["properties"].get("name", "") for _ft in geojson["features"]]
            _name_to_id = {n: i for i, n in enumerate(_geo_names)}
            # Try Thai-name direct match first
            _prov_to_fid = {p: _name_to_id[p] for p in map_df["province"] if p in _name_to_id}
            if len(_prov_to_fid) < 10:
                # GeoJSON likely uses English names — fall back to TH→EN mapping
                _en_lower = {n.lower(): i for i, n in enumerate(_geo_names)}
                for _th, _en in _PROV_TH_EN.items():
                    _fid = _en_lower.get(_en.lower())
                    if _fid is not None:
                        _prov_to_fid[_th] = _fid
                # Fuzzy fallback for Bangkok (กรุงเทพมหานคร) - GeoJSON may use various English names
                if "กรุงเทพมหานคร" not in _prov_to_fid:
                    _bkk_kw = ["bangkok", "krung thep", "krungthep", "metropolis"]
                    for _gn_lo, _gf in _en_lower.items():
                        if any(kw in _gn_lo for kw in _bkk_kw):
                            _prov_to_fid["กรุงเทพมหานคร"] = _gf
                            break
            _plot_df = map_df.copy()
            _plot_df["_fid"] = _plot_df["province"].map(_prov_to_fid)
            _plot_df = _plot_df.dropna(subset=["_fid"])
            _plot_df["_fid"] = _plot_df["_fid"].astype(int)

        if geojson is not None and len(_plot_df) > 0:
            _hover_text = _plot_df.apply(
                lambda r: f"<b>{r['province']}</b><br>{r['n_hospitals']} โรงพยาบาล<br>เขตสุขภาพ {r['health_region']}",
                axis=1,
            ).tolist()
            fig_map = go.Figure(go.Choroplethmapbox(
                geojson=geojson,
                locations=_plot_df["_fid"].tolist(),
                z=_plot_df["n_hospitals"].tolist(),
                featureidkey="id",
                colorscale=[[0, "#fff9c4"], [0.5, "#ffb300"], [1, "#e65100"]],
                zmin=0,
                zmax=int(map_df["n_hospitals"].max()),
                colorbar=dict(title="จำนวน รพ.", len=0.45, x=1.01),
                marker_opacity=0.65,
                marker_line_width=0.5,
                marker_line_color="white",
                text=_hover_text,
                hoverinfo="text",
                name="จำนวน รพ. (ระดับจังหวัด)",
            ))
            if hosp_pts is not None:
                fig_map.add_trace(go.Scattermapbox(
                    lat=hosp_pts["lat_j"], lon=hosp_pts["lon_j"],
                    mode="markers",
                    marker=dict(size=5, color=PINK, opacity=0.5),
                    text=hosp_pts["hover"],
                    hoverinfo="text",
                    name="โรงพยาบาลรายแห่ง",
                ))
            fig_map.update_layout(
                title="การกระจายตัวของโรงพยาบาล 77 จังหวัด",
                height=620,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Sarabun, sans-serif", size=12),
                margin=dict(l=0, r=0, t=50, b=0),
                mapbox=dict(style="carto-positron", zoom=4.5, center=dict(lat=13.0, lon=101.5)),
            )
        else:
            # GeoJSON unavailable or no matches — geo scatter bubble fallback
            fig_map = go.Figure()
            fig_map.add_trace(go.Scattergeo(
                lat=map_df["lat"], lon=map_df["lon"],
                mode="markers",
                marker=dict(
                    size=map_df["n_hospitals"] ** 0.6 * 4,
                    color=map_df["n_hospitals"],
                    colorscale=[[0, "#fff9c4"], [0.5, "#ffb300"], [1, "#e65100"]],
                    colorbar=dict(title="จำนวน รพ.", len=0.5, x=1.02),
                    sizemode="diameter", opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=map_df.apply(
                    lambda r: f"<b>{r['province']}</b><br>{r['n_hospitals']} โรงพยาบาล<br>เขตสุขภาพ {r['health_region']}",
                    axis=1,
                ),
                hoverinfo="text", name="ความหนาแน่น รพ.",
            ))
            if hosp_pts is not None:
                fig_map.add_trace(go.Scattergeo(
                    lat=hosp_pts["lat_j"], lon=hosp_pts["lon_j"],
                    mode="markers",
                    marker=dict(size=4, color=PINK, opacity=0.45, symbol="circle"),
                    text=hosp_pts["hover"], hoverinfo="text", name="โรงพยาบาลรายแห่ง",
                ))
            fig_map.update_layout(
                title="การกระจายตัวของโรงพยาบาล 77 จังหวัด",
                geo=dict(scope="asia", center=dict(lat=13.0, lon=101.5), projection_scale=4.8,
                         showland=True, landcolor="#f8f9fa", showocean=True, oceancolor="#e8f4f8",
                         showcoastlines=True, coastlinecolor="#ccc", showcountries=True,
                         countrycolor="#ddd", showlakes=False, bgcolor="rgba(0,0,0,0)"),
                height=620,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Sarabun, sans-serif", size=12),
                margin=dict(l=0, r=0, t=50, b=0),
            )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("\U0001f7e1 \u0e2a\u0e35\u0e40\u0e02\u0e49\u0e21 (\u0e2a\u0e49\u0e21/\u0e2a\u0e35\u0e40\u0e02\u0e49\u0e21) = \u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14\u0e17\u0e35\u0e48\u0e21\u0e35 \u0e23\u0e1e. \u0e21\u0e32\u0e01 \xb7 \u0e2a\u0e35\u0e2d\u0e48\u0e2d\u0e19 (\u0e40\u0e2b\u0e25\u0e37\u0e2d\u0e07\u0e2d\u0e48\u0e2d\u0e19) = \u0e19\u0e49\u0e2d\u0e22 \xb7 \U0001f534 \u0e08\u0e38\u0e14\u0e40\u0e25\u0e47\u0e01 = \u0e42\u0e23\u0e07\u0e1e\u0e22\u0e32\u0e1a\u0e32\u0e25\u0e23\u0e32\u0e22\u0e41\u0e2b\u0e48\u0e07 (hover \u0e40\u0e1e\u0e37\u0e48\u0e2d\u0e14\u0e39\u0e0a\u0e37\u0e48\u0e2d \u0e1b\u0e23\u0e30\u0e40\u0e20\u0e17 \u0e08\u0e33\u0e19\u0e27\u0e19\u0e40\u0e15\u0e35\u0e22\u0e07)")

        st.markdown("---")
        col_tbl1, col_tbl2 = st.columns(2)
        with col_tbl1:
            st.markdown("**\u0e08\u0e33\u0e19\u0e27\u0e19 \u0e23\u0e1e. \u0e15\u0e48\u0e2d\u0e40\u0e02\u0e15\u0e2a\u0e38\u0e02\u0e20\u0e32\u0e1e**")
            region_summary = (
                hosp_df.groupby("health_region")["n_hospitals"].sum().reset_index()
                .sort_values("n_hospitals", ascending=False)
                .rename(columns={"health_region": "\u0e40\u0e02\u0e15\u0e2a\u0e38\u0e02\u0e20\u0e32\u0e1e", "n_hospitals": "\u0e08\u0e33\u0e19\u0e27\u0e19 \u0e23\u0e1e."})
            )
            st.dataframe(region_summary, use_container_width=True, hide_index=True)
        with col_tbl2:
            st.markdown("**Top 10 \u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14\u0e17\u0e35\u0e48\u0e21\u0e35 \u0e23\u0e1e. \u0e21\u0e32\u0e01\u0e17\u0e35\u0e48\u0e2a\u0e38\u0e14**")
            top10 = hosp_df.nlargest(10, "n_hospitals")[["province", "health_region", "n_hospitals"]].copy()
            top10.columns = ["\u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14", "\u0e40\u0e02\u0e15\u0e2a\u0e38\u0e02\u0e20\u0e32\u0e1e", "\u0e08\u0e33\u0e19\u0e27\u0e19 \u0e23\u0e1e."]
            st.dataframe(top10, use_container_width=True, hide_index=True)


# ==========================================================================
# Tab 4 — User Analytics
# ==========================================================================
with tab4:
    st.markdown("### \U0001f4ca Real-Time User Analytics")
    st.caption("\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e08\u0e32\u0e01 MongoDB Atlas \xb7 collection `ai_sessions` \xb7 \u0e2d\u0e31\u0e1b\u0e40\u0e14\u0e15\u0e17\u0e38\u0e01 5 \u0e19\u0e32\u0e17\u0e35")

    with st.spinner("\u0e01\u0e33\u0e25\u0e31\u0e07\u0e42\u0e2b\u0e25\u0e14\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e08\u0e32\u0e01 MongoDB..."):
        analytics = _load_ai_sessions_analytics()

    status = analytics.get("status")

    if status == "unavailable":
        st.warning("\u26a0\ufe0f \u0e44\u0e21\u0e48\u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e40\u0e0a\u0e37\u0e48\u0e2d\u0e21\u0e15\u0e48\u0e2d MongoDB \u0e44\u0e14\u0e49 \u2014 \u0e15\u0e23\u0e27\u0e08\u0e2a\u0e2d\u0e1a Streamlit Secrets \u0e41\u0e25\u0e30 Network Access")
        st.info("\u0e40\u0e21\u0e37\u0e48\u0e2d app \u0e21\u0e35\u0e1c\u0e39\u0e49\u0e43\u0e0a\u0e49\u0e08\u0e23\u0e34\u0e07 \xb7 \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25 session \u0e08\u0e30\u0e1b\u0e23\u0e32\u0e01\u0e0f\u0e17\u0e35\u0e48\u0e19\u0e35\u0e48\u0e42\u0e14\u0e22\u0e2d\u0e31\u0e15\u0e42\u0e19\u0e21\u0e31\u0e15\u0e34")
    elif status == "empty":
        st.info("\u2139\ufe0f \u0e22\u0e31\u0e07\u0e44\u0e21\u0e48\u0e21\u0e35\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25 session \u0e43\u0e19 MongoDB \xb7 \u0e25\u0e2d\u0e07\u0e43\u0e0a\u0e49 AI Mode \u0e41\u0e25\u0e49\u0e27\u0e01\u0e25\u0e31\u0e1a\u0e21\u0e32\u0e14\u0e39\u0e17\u0e35\u0e48\u0e19\u0e35\u0e48\u0e04\u0e23\u0e31\u0e1a")
    elif status == "error":
        st.error(f"\u0e40\u0e01\u0e34\u0e14\u0e02\u0e49\u0e2d\u0e1c\u0e34\u0e14\u0e1e\u0e25\u0e32\u0e14: {analytics.get('msg', 'unknown')}")
    else:
        total        = analytics["total"]
        top_diseases = analytics["top_diseases"]
        top_symptoms = analytics["top_symptoms"]
        conf_dist    = analytics["confidence_dist"]
        daily        = analytics["daily_trend"]

        def _kpi_card(col, icon, label, value, sub):
            col.markdown(
                f'<div style="background:rgba(27,154,170,0.08);border-radius:10px;padding:14px 16px;text-align:center;">' +
                f'<div style="font-size:22px">{icon}</div>' +
                f'<div style="font-size:12px;color:#888;margin:4px 0">{label}</div>' +
                f'<div style="font-size:1.4rem;font-weight:700;color:#0a2342">{value}</div>' +
                f'<div style="font-size:11px;color:#555;margin-top:4px;line-height:1.4">{sub}</div>' +
                '</div>',
                unsafe_allow_html=True,
            )

        k1, k2, k3, k4 = st.columns(4)
        _kpi_card(k1, "\U0001f50d", "Sessions \u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14", f"{total:,}", "\u0e19\u0e31\u0e1a\u0e15\u0e31\u0e49\u0e07\u0e41\u0e15\u0e48\u0e40\u0e1b\u0e34\u0e14\u0e15\u0e31\u0e27")
        top_dis_val = top_diseases.iloc[0]["disease"] if not top_diseases.empty else "\u2014"
        _kpi_card(k2, "\U0001f3c6", "\u0e42\u0e23\u0e04\u0e17\u0e35\u0e48\u0e04\u0e49\u0e19\u0e1a\u0e48\u0e2d\u0e22\u0e2a\u0e38\u0e14", top_dis_val, "\u0e08\u0e32\u0e01 AI Mode")
        top_sym_val = top_symptoms.iloc[0]["symptom"] if not top_symptoms.empty else "\u2014"
        _kpi_card(k3, "\U0001fa7a", "\u0e2d\u0e32\u0e01\u0e32\u0e23\u0e17\u0e35\u0e48\u0e1e\u0e1a\u0e1a\u0e48\u0e2d\u0e22\u0e2a\u0e38\u0e14", top_sym_val, "\u0e23\u0e27\u0e21\u0e17\u0e38\u0e01 session")
        if not conf_dist.empty:
            dom = conf_dist.sort_values("n", ascending=False).iloc[0]
            conf_pct = int(dom["n"] / conf_dist["n"].sum() * 100)
            _kpi_card(k4, "\U0001f4c8", "Confidence \u0e17\u0e35\u0e48\u0e1e\u0e1a\u0e1a\u0e48\u0e2d\u0e22", dom["label"], f"{conf_pct}% \u0e02\u0e2d\u0e07\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14")

        st.markdown("---")
        row1_l, row1_r = st.columns(2)
        with row1_l:
            if not top_diseases.empty:
                fig_td = px.bar(
                    top_diseases.sort_values("n"),
                    x="n", y="disease", orientation="h",
                    color="n",
                    color_continuous_scale=[[0, "#ffecd2"], [1, PINK]],
                    title="\U0001f3c6 Top \u0e42\u0e23\u0e04\u0e17\u0e35\u0e48 User \u0e04\u0e49\u0e19\u0e2b\u0e32\u0e1a\u0e48\u0e2d\u0e22\u0e17\u0e35\u0e48\u0e2a\u0e38\u0e14",
                    labels={"n": "\u0e08\u0e33\u0e19\u0e27\u0e19 sessions", "disease": "\u0e42\u0e23\u0e04"},
                )
                fig_td.update_coloraxes(showscale=False)
                st.plotly_chart(_fig_style(fig_td), use_container_width=True)
        with row1_r:
            if not conf_dist.empty:
                conf_colors = {"\u0e21\u0e31\u0e48\u0e19\u0e43\u0e08\u0e2a\u0e39\u0e07": GREEN,
                               "\u0e21\u0e31\u0e48\u0e19\u0e43\u0e08\u0e1b\u0e32\u0e19\u0e01\u0e25\u0e32\u0e07": TEAL,
                               "\u0e21\u0e31\u0e48\u0e19\u0e43\u0e08\u0e15\u0e48\u0e33": AMBER,
                               "\u0e21\u0e31\u0e48\u0e19\u0e43\u0e08\u0e15\u0e48\u0e33\u0e21\u0e32\u0e01": PINK}
                fig_conf = px.pie(conf_dist, names="label", values="n",
                    title="\U0001f4c8 Confidence Level Distribution",
                    color="label", color_discrete_map=conf_colors, hole=0.4)
                fig_conf.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(_fig_style(fig_conf), use_container_width=True)

        row2_l, row2_r = st.columns(2)
        with row2_l:
            if not top_symptoms.empty:
                fig_sym = px.bar(
                    top_symptoms.sort_values("n"),
                    x="n", y="symptom", orientation="h",
                    color="n",
                    color_continuous_scale=[[0, "#c7ecf7"], [1, NAVY]],
                    title="\U0001fa7a Top \u0e2d\u0e32\u0e01\u0e32\u0e23\u0e17\u0e35\u0e48 User \u0e23\u0e32\u0e22\u0e07\u0e32\u0e19\u0e1a\u0e48\u0e2d\u0e22\u0e2a\u0e38\u0e14",
                    labels={"n": "\u0e08\u0e33\u0e19\u0e27\u0e19\u0e04\u0e23\u0e31\u0e49\u0e07", "symptom": "\u0e2d\u0e32\u0e01\u0e32\u0e23 (symptom_en)"},
                )
                fig_sym.update_coloraxes(showscale=False)
                st.plotly_chart(_fig_style(fig_sym), use_container_width=True)
        with row2_r:
            if not daily.empty and len(daily) > 1:
                fig_trend = px.line(daily, x="date", y="sessions",
                    title="\U0001f4c5 \u0e08\u0e33\u0e19\u0e27\u0e19 Sessions \u0e15\u0e48\u0e2d\u0e27\u0e31\u0e19",
                    labels={"date": "\u0e27\u0e31\u0e19\u0e17\u0e35\u0e48", "sessions": "\u0e08\u0e33\u0e19\u0e27\u0e19 sessions"},
                    color_discrete_sequence=[TEAL], markers=True)
                fig_trend.update_traces(line_width=2.5)
                st.plotly_chart(_fig_style(fig_trend), use_container_width=True)
            elif not daily.empty:
                st.info("\u0e15\u0e49\u0e2d\u0e07\u0e01\u0e32\u0e23\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e21\u0e32\u0e01\u0e01\u0e27\u0e48\u0e32 1 \u0e27\u0e31\u0e19\u0e40\u0e1e\u0e37\u0e48\u0e2d\u0e41\u0e2a\u0e14\u0e07 trend")

        st.caption("\U0001f504 \u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e19\u0e35\u0e49\u0e14\u0e36\u0e07\u0e08\u0e32\u0e01 MongoDB Atlas (Cloud) \xb7 refresh \u0e2d\u0e31\u0e15\u0e42\u0e19\u0e21\u0e31\u0e15\u0e34\u0e17\u0e38\u0e01 5 \u0e19\u0e32\u0e17\u0e35 \xb7 \u0e2a\u0e30\u0e17\u0e49\u0e2d\u0e19\u0e1e\u0e24\u0e15\u0e34\u0e01\u0e23\u0e23\u0e21 user \u0e08\u0e23\u0e34\u0e07\u0e02\u0e2d\u0e07 app")
