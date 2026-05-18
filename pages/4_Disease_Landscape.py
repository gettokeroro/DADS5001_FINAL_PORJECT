"""
Page 4 · Disease Landscape
แหล่งข้อมูล:
  - ha_aod_001-2.csv  (TIS-620) → top-disease mentions จาก รพ.ทั่วประเทศ
  - disease_specialty_mapping.csv → ความครอบคลุมของ app (48 โรค × 13 สาขา)
  - disease_drug_mapping_v2_ed.csv → drug tier distribution
  - hospitals_thailand.csv       → hospital count per province / health region
  - MongoDB ai_sessions           → real user analytics (top diseases, symptoms, confidence)
"""
import re
import sys
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Disease Landscape", page_icon="🗺️", layout="wide")
init_session_state()
render_disclaimer_sidebar()

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------
NAVY   = "#0a2342"
TEAL   = "#1B9AAA"
PINK   = "#E84C8B"
AMBER  = "#F7B731"
GREEN  = "#26de81"


def _fig_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sarabun, sans-serif", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_hospital_disease_mentions() -> pd.DataFrame:
    """Parse กลุ่มผู้ป่วยโรคสำคัญ free-text from ha_aod_001-2.csv (cp874).
    Returns DataFrame[disease, count] top-25 sorted descending.
    """
    raw_path = DATA / "raw" / "datago" / "ha_aod_001-2.csv"
    try:
        df = pd.read_csv(raw_path, encoding="cp874", low_memory=False, header=0)
        # Last column (index 19) = กลุ่มผู้ป่วยโรคสำคัญ
        disease_col = df.iloc[:, 19].dropna().astype(str)
        counts: Counter = Counter()
        for text in disease_col:
            parts = re.split(r"[,/\\\n;·•]+", text)
            for p in parts:
                p = p.strip().strip('"').strip("'").strip()
                if len(p) >= 2 and p.lower() not in ("nan", "", "-", "null", "none"):
                    counts[p] += 1
        if not counts:
            return pd.DataFrame(columns=["disease", "count"])
        return pd.DataFrame(counts.most_common(25), columns=["disease", "count"])
    except Exception:
        return pd.DataFrame(columns=["disease", "count"])


@st.cache_data(show_spinner=False)
def _load_specialty_counts() -> pd.DataFrame:
    """Count diseases per primary_specialty from disease_specialty_mapping.csv."""
    path = DATA / "processed" / "disease_specialty_mapping.csv"
    try:
        df = pd.read_csv(path)
        return (
            df.groupby("primary_specialty")
            .size()
            .reset_index(name="n_diseases")
            .sort_values("n_diseases", ascending=False)
        )
    except Exception:
        return pd.DataFrame(columns=["primary_specialty", "n_diseases"])


@st.cache_data(show_spinner=False)
def _load_drug_tier_counts() -> pd.DataFrame:
    """Count unique drugs per prescription_tier."""
    path = DATA / "processed" / "disease_drug_mapping_v2_ed.csv"
    try:
        df = pd.read_csv(path)
        counts = (
            df.drop_duplicates(subset=["drug_generic"])
            .groupby("prescription_tier")
            .size()
            .reset_index(name="n_drugs")
            .sort_values("n_drugs", ascending=False)
        )
        label_map = {
            "otc":      "ซื้อเองได้ (OTC)",
            "pharmacy": "เภสัชกรแนะนำ",
            "doctor":   "ต้องใบสั่งแพทย์",
        }
        counts["tier_label"] = counts["prescription_tier"].map(label_map).fillna(counts["prescription_tier"])
        return counts
    except Exception:
        return pd.DataFrame(columns=["prescription_tier", "n_drugs", "tier_label"])


@st.cache_data(show_spinner=False)
def _load_hospital_province() -> pd.DataFrame:
    """Count hospitals per province + health_region."""
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


@st.cache_data(ttl=300, show_spinner=False)
def _load_ai_sessions_analytics() -> dict:
    """Query MongoDB ai_sessions → aggregated analytics. TTL=300 s.
    Returns dict with status key: 'ok' | 'unavailable' | 'empty' | 'error'.
    """
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

        # top diseases
        disease_counts = (
            df["top_disease"].value_counts().head(15)
            .reset_index().rename(columns={"top_disease": "disease", "count": "n"})
        )

        # top symptoms (flatten list)
        all_sym: list = []
        for row in df["symptoms"].dropna():
            if isinstance(row, list):
                all_sym.extend(row)
        symptom_df = (
            pd.DataFrame(Counter(all_sym).most_common(15), columns=["symptom", "n"])
            if all_sym else pd.DataFrame(columns=["symptom", "n"])
        )

        # confidence distribution
        conf_counts = (
            df["confidence_level"].value_counts()
            .reset_index().rename(columns={"confidence_level": "level", "count": "n"})
        )
        lbl = {"high": "มั่นใจสูง", "medium": "มั่นใจปานกลาง",
               "low": "มั่นใจต่ำ", "very_low": "มั่นใจต่ำมาก"}
        conf_counts["label"] = conf_counts["level"].map(lbl).fillna(conf_counts["level"])

        # daily trend
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date").size().reset_index(name="sessions")
        daily["date"] = pd.to_datetime(daily["date"])

        return {
            "status":          "ok",
            "total":           len(df),
            "top_diseases":    disease_counts,
            "top_symptoms":    symptom_df,
            "confidence_dist": conf_counts,
            "daily_trend":     daily,
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🗺️ Disease Landscape Thailand")
st.markdown("##### ภาพรวมโรคในประเทศไทย · ข้อมูลจากโรงพยาบาลจริง + MongoDB User Analytics")
st.divider()

# ---------------------------------------------------------------------------
# External reference stats row (DDC / WHO / IDF 2024–2025)
# ---------------------------------------------------------------------------
with st.expander("📌 สถิติโรคไทย 2024–2025 (อ้างอิงจาก DDC · WHO · IDF)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💀 เสียชีวิตจาก NCDs/ปี", "~420,000 คน", "74% ของการเสียชีวิตทั้งหมด")
    c2.metric("🩸 ผู้ป่วยเบาหวาน", "6.36 ล้านคน", "อันดับ 4 Western Pacific · ปี 2024")
    c3.metric("❤️ ผู้ป่วยความดันโลหิตสูง", "~13.7 ล้านคน", "เพิ่มขึ้นต่อเนื่องทุกปี")
    c4.metric("🏥 ภาระโรค NCDs รวม", "~14 ล้านคน", "เบาหวาน · ความดัน · หัวใจ · ไต")
    st.caption("Source: กรมควบคุมโรค (DDC) · WHO Thailand 2025 · IDF Diabetes Atlas 2024 · สสส. 2568")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 โรคใน รพ.ไทย",
    "🧬 ความครอบคลุม App",
    "📍 แผนที่โรงพยาบาล",
    "📊 User Analytics",
])


# ==========================================================================
# Tab 1 — Hospital disease mentions
# ==========================================================================
with tab1:
    st.markdown("### 🏥 โรคที่โรงพยาบาลไทยรายงานว่าพบบ่อย")
    st.caption("Source: ข้อมูลโรงพยาบาลสังกัดกระทรวงสาธารณสุข (ha_aod_001-2) · กลุ่มผู้ป่วยโรคสำคัญ — 437 รพ. ทั่วประเทศ")

    with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
        mentions_df = _load_hospital_disease_mentions()

    if mentions_df.empty:
        st.warning("ไม่พบข้อมูล — ตรวจสอบไฟล์ `data/raw/datago/ha_aod_001-2.csv`")
    else:
        col_chart, col_stat = st.columns([3, 1])

        with col_chart:
            fig = px.bar(
                mentions_df.sort_values("count"),
                x="count", y="disease", orientation="h",
                color="count",
                color_continuous_scale=[[0, "#c7ecf7"], [1, TEAL]],
                title="Top 25 โรคที่รพ.ไทยรายงานในกลุ่มผู้ป่วยสำคัญ",
                labels={"count": "จำนวนโรงพยาบาลที่รายงาน", "disease": "โรค"},
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis=dict(tickfont=dict(size=12)))
            st.plotly_chart(_fig_style(fig), use_container_width=True)

        with col_stat:
            st.markdown("**Top 5 โรคที่พบบ่อยสุด**")
            max_count = mentions_df["count"].max()
            for _, row in mentions_df.head(5).iterrows():
                pct = int(row["count"] / max_count * 100)
                st.markdown(f"**{row['disease']}**")
                st.progress(pct / 100, text=f"{int(row['count'])} รพ.")
            st.caption(f"วิเคราะห์จาก {mentions_df['count'].sum():,} mentions")

    st.info("💡 ข้อมูลนี้มาจากฟิลด์ 'กลุ่มผู้ป่วยโรคสำคัญ' ที่แต่ละโรงพยาบาลรายงานเอง — สะท้อนโรคที่แต่ละ รพ. เห็นในทางปฏิบัติจริง ไม่ใช่ตัวเลขทางระบาดวิทยาทางการ")


# ==========================================================================
# Tab 2 — App coverage
# ==========================================================================
with tab2:
    st.markdown("### 🧬 ความครอบคลุมของ Triage App")
    st.caption("48 โรค · 13 สาขาแพทย์ · 86 ยาในบัญชียาหลัก — ออกแบบให้ครอบคลุมโรคที่พบบ่อยในประชาชน")

    spec_df = _load_specialty_counts()
    drug_df = _load_drug_tier_counts()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        if not spec_df.empty:
            spec_df["specialty_th"] = (
                spec_df["primary_specialty"]
                .str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
            )
            fig_spec = px.bar(
                spec_df.sort_values("n_diseases"),
                x="n_diseases", y="specialty_th", orientation="h",
                color="n_diseases",
                color_continuous_scale=[[0, "#ffecd2"], [1, PINK]],
                title="จำนวนโรคต่อสาขาแพทย์ (48 โรค)",
                labels={"n_diseases": "จำนวนโรค", "specialty_th": "สาขา"},
            )
            fig_spec.update_coloraxes(showscale=False)
            st.plotly_chart(_fig_style(fig_spec), use_container_width=True)

    with col_r:
        if not drug_df.empty:
            fig_drug = px.pie(
                drug_df, names="tier_label", values="n_drugs",
                title="86 ยา แบ่งตาม Prescription Tier",
                color_discrete_sequence=[GREEN, TEAL, AMBER],
                hole=0.45,
            )
            fig_drug.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(_fig_style(fig_drug), use_container_width=True)

    # ICD-10 treemap
    st.markdown("---")
    st.markdown("**ICD-10 Chapter Distribution — 48 โรคใน App**")
    try:
        icd_df = pd.read_csv(DATA / "processed" / "disease_specialty_mapping.csv")
        icd_counts = (
            icd_df.groupby("icd10_chapter_name").size()
            .reset_index(name="n").sort_values("n", ascending=False)
        )
        fig_icd = px.treemap(
            icd_counts, path=["icd10_chapter_name"], values="n",
            color="n",
            color_continuous_scale=[[0, "#dfe6e9"], [1, NAVY]],
            title="48 โรค จัดกลุ่มตาม ICD-10 Chapter",
        )
        fig_icd.update_coloraxes(showscale=False)
        fig_icd.update_traces(textinfo="label+value")
        st.plotly_chart(_fig_style(fig_icd), use_container_width=True)
    except Exception:
        st.info("ไม่สามารถโหลด ICD-10 chapter data ได้")


# ==========================================================================
# Tab 3 — Hospital map (treemap + bar by health region)
# ==========================================================================
with tab3:
    st.markdown("### 📍 การกระจายตัวของโรงพยาบาลทั่วประเทศ")
    st.caption("Source: hospitals_thailand.csv · 1,581 โรงพยาบาล · 77 จังหวัด · 13 เขตสุขภาพ")

    hosp_df = _load_hospital_province()

    if hosp_df.empty:
        st.warning("ไม่พบข้อมูล hospitals_thailand.csv")
    else:
        col_tree, col_bar = st.columns([3, 2])

        with col_tree:
            fig_tree = px.treemap(
                hosp_df, path=["health_region", "province"], values="n_hospitals",
                color="n_hospitals",
                color_continuous_scale=[[0, "#c7ecf7"], [1, NAVY]],
                title="รพ. จัดกลุ่มตาม เขตสุขภาพ → จังหวัด",
            )
            fig_tree.update_coloraxes(showscale=False)
            fig_tree.update_traces(textinfo="label+value")
            st.plotly_chart(_fig_style(fig_tree), use_container_width=True)

        with col_bar:
            region_summary = (
                hosp_df.groupby("health_region")["n_hospitals"]
                .sum().reset_index()
                .sort_values("n_hospitals", ascending=True)
            )
            fig_reg = px.bar(
                region_summary,
                x="n_hospitals", y="health_region", orientation="h",
                color="n_hospitals",
                color_continuous_scale=[[0, "#dfe6e9"], [1, TEAL]],
                title="จำนวน รพ. ต่อเขตสุขภาพ",
                labels={"n_hospitals": "จำนวน รพ.", "health_region": "เขตสุขภาพ"},
            )
            fig_reg.update_coloraxes(showscale=False)
            st.plotly_chart(_fig_style(fig_reg), use_container_width=True)

        st.markdown("**Top 10 จังหวัดที่มี รพ.มากที่สุด**")
        top10 = hosp_df.nlargest(10, "n_hospitals")[["province", "health_region", "n_hospitals"]].copy()
        top10.columns = ["จังหวัด", "เขตสุขภาพ", "จำนวน รพ."]
        st.dataframe(top10, use_container_width=True, hide_index=True)


# ==========================================================================
# Tab 4 — MongoDB User Analytics
# ==========================================================================
with tab4:
    st.markdown("### 📊 Real-Time User Analytics")
    st.caption("ข้อมูลจาก MongoDB Atlas · collection `ai_sessions` · อัปเดตทุก 5 นาที · บันทึกทุกครั้งที่ผู้ใช้วินิจฉัยใน AI Mode")

    with st.spinner("กำลังโหลดข้อมูลจาก MongoDB..."):
        analytics = _load_ai_sessions_analytics()

    status = analytics.get("status")

    if status == "unavailable":
        st.warning("⚠️ ไม่สามารถเชื่อมต่อ MongoDB ได้ — ตรวจสอบ Streamlit Secrets และ Network Access")
        st.info("เมื่อ app มีผู้ใช้จริง · ข้อมูล session จะปรากฏที่นี่โดยอัตโนมัติ")
    elif status == "empty":
        st.info("ℹ️ ยังไม่มีข้อมูล session ใน MongoDB · ลองใช้ AI Mode แล้วกลับมาดูที่นี่ครับ")
    elif status == "error":
        st.error(f"เกิดข้อผิดพลาด: {analytics.get('msg', 'unknown')}")
    else:
        total        = analytics["total"]
        top_diseases = analytics["top_diseases"]
        top_symptoms = analytics["top_symptoms"]
        conf_dist    = analytics["confidence_dist"]
        daily        = analytics["daily_trend"]

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔍 Sessions ทั้งหมด", f"{total:,}")
        if not top_diseases.empty:
            k2.metric("🏆 โรคที่ค้นบ่อยสุด", top_diseases.iloc[0]["disease"])
        if not top_symptoms.empty:
            k3.metric("🩺 อาการที่พบบ่อยสุด", top_symptoms.iloc[0]["symptom"])
        if not conf_dist.empty:
            dom = conf_dist.sort_values("n", ascending=False).iloc[0]
            k4.metric("📈 Confidence ที่พบบ่อย", dom["label"])

        st.markdown("---")
        row1_l, row1_r = st.columns(2)

        with row1_l:
            if not top_diseases.empty:
                fig_td = px.bar(
                    top_diseases.sort_values("n"),
                    x="n", y="disease", orientation="h",
                    color="n",
                    color_continuous_scale=[[0, "#ffecd2"], [1, PINK]],
                    title="🏆 Top โรคที่ User ค้นหาบ่อยที่สุด",
                    labels={"n": "จำนวน sessions", "disease": "โรค"},
                )
                fig_td.update_coloraxes(showscale=False)
                st.plotly_chart(_fig_style(fig_td), use_container_width=True)

        with row1_r:
            if not conf_dist.empty:
                conf_colors = {
                    "มั่นใจสูง":      GREEN,
                    "มั่นใจปานกลาง": TEAL,
                    "มั่นใจต่ำ":      AMBER,
                    "มั่นใจต่ำมาก":  PINK,
                }
                fig_conf = px.pie(
                    conf_dist, names="label", values="n",
                    title="📈 Confidence Level Distribution",
                    color="label", color_discrete_map=conf_colors,
                    hole=0.4,
                )
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
                    title="🩺 Top อาการที่ User รายงานบ่อยสุด",
                    labels={"n": "จำนวนครั้ง", "symptom": "อาการ (symptom_en)"},
                )
                fig_sym.update_coloraxes(showscale=False)
                st.plotly_chart(_fig_style(fig_sym), use_container_width=True)

        with row2_r:
            if not daily.empty and len(daily) > 1:
                fig_trend = px.line(
                    daily, x="date", y="sessions",
                    title="📅 จำนวน Sessions ต่อวัน",
                    labels={"date": "วันที่", "sessions": "จำนวน sessions"},
                    color_discrete_sequence=[TEAL],
                    markers=True,
                )
                fig_trend.update_traces(line_width=2.5)
                st.plotly_chart(_fig_style(fig_trend), use_container_width=True)
            elif not daily.empty:
                st.info("ต้องการข้อมูลมากกว่า 1 วันเพื่อแสดง trend · ยังคงสะสมข้อมูลอยู่")

        st.caption("🔄 ข้อมูลนี้ดึงจาก MongoDB Atlas (Cloud) · refresh อัตโนมัติทุก 5 นาที · สะท้อนพฤติกรรม user จริงของ app")
