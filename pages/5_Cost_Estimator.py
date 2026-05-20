"""
Page 5 · คู่มือการใช้งาน — Symptom-to-Specialty Triage
แปลงจาก Cost Estimator stub · DADS5001 Final Project
"""
import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar

st.set_page_config(page_title="คู่มือการใช้งาน", page_icon="📖", layout="wide")
init_session_state()
render_disclaimer_sidebar()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("📖 คู่มือการใช้งาน")
st.caption("Symptom-to-Specialty Triage · DADS5001 Final Project")

_c1, _c2, _c3 = st.columns(3)
_c1.metric("โรคในฐานข้อมูล", "48 โรค")
_c2.metric("อาการที่รองรับ", "132 อาการ")
_c3.metric("สาขาแพทย์", "13 สาขา")

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 1 · ภาพรวม
# ─────────────────────────────────────────────────────────────
st.markdown("## 🏥 ส่วนที่ 1 · ภาพรวม App")
st.markdown(
    "**Symptom-to-Specialty Triage** ช่วยคนไทยประเมินอาการเบื้องต้นก่อนตัดสินใจไปโรงพยาบาล "
    "โดยแนะนำ **สาขาแพทย์ที่เหมาะสม** และ **ระดับความเร่งด่วน** "
    "(1–5 ตามมาตรฐาน ED Triage กระทรวงสาธารณสุข)\n\n"
    "> ⚠️ **แอปนี้ไม่วินิจฉัยโรค และไม่แทนที่การพบแพทย์จริง**"
)

# Mermaid flowchart — ใช้ components.html (ไม่ถูกจำกัดด้วย Markdown single-line rule)
_mermaid_html = (
    '<div style="text-align:center;padding:8px;">' +
    '<div class="mermaid">flowchart LR\n' +
    '    A["🤒 มีอาการ"] --> B{"เลือก Mode"}\n' +
    '    B --> C["📋 Non-AI Mode"]\n' +
    '    B --> D["🤖 AI Mode (น้องอุ่นใน)"]\n' +
    '    C --> E["✅ เลือก checkbox\\n132 อาการ"]\n' +
    '    D --> F["💬 พิมพ์/เลือก\\nอาการภาษาไทย"]\n' +
    '    E --> G["📊 วิเคราะห์"]\n' +
    '    F --> G\n' +
    '    G --> H["🎯 สาขาแพทย์\\n+ ระดับเร่งด่วน"]\n' +
    '    H --> I["💊 ยา & 🏥 รพ."]\n' +
    '    I --> J["🩺 พบแพทย์"]\n' +
    '</div></div>' +
    '<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>' +
    '<script>mermaid.initialize({startOnLoad:true,theme:"default",flowchart:{curve:"basis"}});</script>'
)
components.html(_mermaid_html, height=230)

st.markdown("### เลือก Mode ไหนดี?")
st.markdown(
    "| | 📋 Non-AI Mode | 🤖 AI Mode (น้องอุ่นใน) |\n"
    "|---|---|---|\n"
    "| **วิธีระบุอาการ** | tick checkbox จากรายการ | พิมพ์หรือเลือกเป็นภาษาไทย |\n"
    "| **เหมาะกับ** | รู้ว่าตัวเองมีอาการอะไรแน่ชัด | อยากเล่าอาการแบบธรรมชาติ |\n"
    "| **ความเร็ว** | เร็วมาก (ไม่ใช้ AI) | ช้ากว่าเล็กน้อย (เรียก LLM) |\n"
    "| **Internet** | ไม่ต้องการ | ต้องการเชื่อมต่อ |\n"
    "| **Rate limit** | ไม่มี | 20 ครั้ง/session |\n"
    "| **การวิเคราะห์** | TF-IDF + Bayes scoring | LLM extract + TF-IDF scoring |"
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 2 · Non-AI Mode
# ─────────────────────────────────────────────────────────────
st.markdown("## 📋 ส่วนที่ 2 · Non-AI Mode")
st.info("**เหมาะสำหรับ:** ผู้ที่รู้ชัดว่ามีอาการอะไร และต้องการผลรวดเร็วโดยไม่ใช้ AI")

_steps_nonai = [
    ("1️⃣", "เปิดหน้า **Non-AI Mode** จาก sidebar ด้านซ้าย"),
    (
        "2️⃣",
        "**เลือกอาการ** จาก checkbox ที่จัดกลุ่มตาม body system "
        "— ใช้ช่องค้นหาด้านบนพิมพ์ชื่ออาการภาษาไทยเพื่อกรอง "
        "· เลือกได้หลายอาการพร้อมกัน · ยิ่งเลือกครบ ผลลัพธ์ยิ่งแม่นยำ",
    ),
    ("3️⃣", "**เลือกจังหวัด** (ไม่บังคับ) เพื่อกรองโรงพยาบาลในพื้นที่ที่สะดวก"),
    ("4️⃣", "กดปุ่ม **'🔍 วิเคราะห์อาการ'** เพื่อรับผลลัพธ์"),
    (
        "5️⃣",
        "**อ่านผลลัพธ์:** Confidence badge (สูง 🟢 / กลาง 🟡 / ต่ำ 🟠 / ต่ำมาก 🔴) "
        "· Top-3 โรคที่น่าจะเป็น · สาขาแพทย์แนะนำ · ระดับเร่งด่วน 1–5 "
        "· Expander ยาที่เกี่ยวข้อง + โรงพยาบาล",
    ),
]
for _icon, _text in _steps_nonai:
    st.markdown(f"**{_icon}** {_text}")

st.info(
    "💡 **เคล็ดลับ:** ถ้า confidence badge แสดง 🔴 ต่ำมาก "
    "ลองเพิ่มอาการ หรือใช้ AI Mode ที่น้องอุ่นในจะถามเพิ่มให้เอง"
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 3 · AI Mode
# ─────────────────────────────────────────────────────────────
st.markdown("## 🤖 ส่วนที่ 3 · AI Mode — น้องอุ่นใน")

# mascot card — single-line HTML (per Streamlit markdown HTML rule)
st.markdown('<div style="background:linear-gradient(135deg,#fff5f5 0%,#ffe4e4 100%);border-radius:16px;padding:20px 24px;border-left:5px solid #ff8fa3;margin-bottom:16px;"><h4 style="margin:0 0 8px 0;">🌸 น้องอุ่นใน (Aoon-nai) คือใคร?</h4><p style="margin:0;color:#555;">น้องอุ่นใน คือ AI ผู้ช่วยของ Symptom-to-Specialty Triage มีน้ำเสียงอบอุ่น เรียกผู้ใช้ว่า "พี่" และจะถามคำถามเพิ่มเติมแบบ adaptive เพื่อเก็บข้อมูลอาการให้ครบก่อนส่งผลวิเคราะห์</p></div>', unsafe_allow_html=True)

st.info("**เหมาะสำหรับ:** ผู้ที่อยากเล่าอาการแบบภาษาธรรมชาติ หรือยังไม่แน่ใจว่ามีอาการอะไรบ้าง")

_steps_ai = [
    ("1️⃣", "เปิดหน้า **AI Mode** จาก sidebar ด้านซ้าย"),
    (
        "2️⃣",
        "**เลือก Chief Complaint** (อาการหลัก) จาก 6 หมวด: "
        "เป็นไข้ · ไข้หวัด · ปวดหัว · ปวดท้อง · ปวดกล้ามเนื้อ · ผื่น "
        "(หรือพิมพ์อาการอื่นในช่อง free text)",
    ),
    (
        "3️⃣",
        "**ตอบคำถามน้องอุ่นใน** — น้องจะถามแบบ adaptive สูงสุด ~6 ข้อ "
        "เช่น ระยะเวลา · ความรุนแรง · อาการร่วม · ปัจจัยเสี่ยง",
    ),
    (
        "4️⃣",
        "**อ่านผลลัพธ์:** คำบรรยายจากน้องอุ่นในพร้อม reasoning "
        "· Confidence badge · Top-3 โรคที่น่าจะเป็น + สาขาแพทย์แนะนำ "
        "· แจ้งเตือน red-flag symptoms ถ้ามี",
    ),
    ("5️⃣", "**เลือกจังหวัด** เพื่อกรองโรงพยาบาล และดู Expander ยา"),
]
for _icon, _text in _steps_ai:
    st.markdown(f"**{_icon}** {_text}")

st.warning(
    "⚠️ **Rate limit:** AI Mode จำกัด **20 ครั้งต่อ session** เพื่อควบคุมค่าใช้จ่าย API "
    "หากเกินขีดจำกัด ให้รีเฟรชหน้าเพื่อเริ่ม session ใหม่"
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 4 · ข้อจำกัดและคำเตือน
# ─────────────────────────────────────────────────────────────
st.markdown("## ⚠️ ส่วนที่ 4 · ข้อจำกัดและคำเตือนด้านการแพทย์")

st.error(
    "🚨 **อาการฉุกเฉิน — โทร 1669 ทันที หรือไปห้องฉุกเฉินโดยตรง**\n\n"
    "เช่น หัวใจวาย · หายใจไม่ออก · ไม่รู้สึกตัว · เจ็บหน้าอกรุนแรง · อัมพาต/พูดไม่ชัดฉับพลัน (FAST)"
)

_warnings = [
    ("🔴", "**ไม่วินิจฉัยโรค** — ผลลัพธ์คือ _การแนะนำสาขาแพทย์_ เท่านั้น ไม่ใช่การวินิจฉัยจากแพทย์จริง"),
    ("🔴", "**ไม่แทนแพทย์** — ข้อมูลที่ได้เป็นเพียงการประเมินเบื้องต้น ควรพบแพทย์เพื่อรับการวินิจฉัยที่ถูกต้อง"),
    (
        "🟡",
        "**Confidence ต่ำมาก 🔴** — หากผลลัพธ์แสดง confidence ต่ำมาก "
        "แนะนำให้พบแพทย์โดยตรง ไม่ควรตัดสินใจจากผลนี้เพียงอย่างเดียว",
    ),
    (
        "🟡",
        "**ยาในระบบ** — รายการยาที่แสดงเป็นข้อมูลอ้างอิงทางการแพทย์ทั่วไป "
        "ไม่ใช่ใบสั่งยา ห้ามซื้อยาเองโดยไม่ปรึกษาเภสัชกรหรือแพทย์",
    ),
    (
        "🟡",
        "**โรคหายาก / โรคซับซ้อน** — ฐานข้อมูล 48 โรคครอบคลุมโรคพบบ่อยในไทย "
        "โรคหายากหรือซับซ้อนอาจไม่อยู่ในระบบ",
    ),
    ("🟢", "**ความเป็นส่วนตัว** — session log บันทึกลง MongoDB Atlas แบบ anonymized ไม่มีข้อมูลส่วนตัวของผู้ใช้"),
]
for _color, _text in _warnings:
    st.markdown(f"{_color} {_text}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 5 · แหล่งข้อมูล
# ─────────────────────────────────────────────────────────────
st.markdown("## 📚 ส่วนที่ 5 · แหล่งข้อมูล")

with st.expander("🗃️ Symptom-Disease Backbone Dataset"):
    st.markdown(
        "- **[itachi9604/disease-symptom-description-dataset]"
        "(https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)** (Kaggle)\n"
        "  - 48 โรค × 132 อาการ (binary matrix) หลัง augment เพิ่ม 7 โรคพบบ่อยในไทย\n"
        "  - รองรับ TF-IDF scoring + Naive Bayes classifier"
    )

with st.expander("🏥 มาตรฐานการแพทย์ไทย"):
    st.markdown(
        "- **ICD-10-TM** — Thai Modification ของ International Classification of Diseases "
        "(กรมการแพทย์ กระทรวงสาธารณสุข)\n"
        "- **ED Triage MOPH** — แนวทาง 5-level Triage มาตรฐานห้องฉุกเฉินไทย\n"
        "- **disease_specialty_mapping.csv** — สร้างเอง: mapping 48 โรค × 13 สาขาแพทย์"
    )

with st.expander("📊 สถิติสาธารณสุขไทย (data.go.th)"):
    st.markdown(
        "- การจัดอันดับโรคของผู้ป่วยนอก/ใน พ.ศ. 2564\n"
        "- 20 ลำดับโรคสูงสุด IPD 298 กลุ่มโรค พ.ศ. 2559–2566 (time series 8 ปี)\n"
        "- ผู้ป่วยนอก 21 กลุ่มโรค พ.ศ. 2559–2565\n"
        "- การเข้ารับบริการ OPD พ.ศ. 2566\n"
        "- โรคติดต่อ/เฝ้าระวัง พ.ศ. 2567\n"
        "- ประชากรตามอายุ สิทธิหลักประกันสุขภาพ พ.ศ. 2565\n"
        "- ข้อมูลทั่วไปของโรงพยาบาล พ.ศ. 2566"
    )

with st.expander("☁️ Cloud Storage"):
    st.markdown(
        "- **Snowflake** (AWS ap-southeast-1) — เก็บ training data, drug mapping, hospital list\n"
        "- **MongoDB Atlas** (AWS ap-southeast-1, M0 Free) "
        "— เก็บ symptom dictionary, disease prevalence, AI session log"
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# Download HTML
# ─────────────────────────────────────────────────────────────
st.markdown("## 📥 ดาวน์โหลดคู่มือ")
st.markdown(
    "บันทึกคู่มือเป็นไฟล์ HTML → เปิดในเบราว์เซอร์ → **Ctrl+P** → **Save as PDF**"
)

_HTML_MANUAL = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>คู่มือการใช้งาน — Symptom-to-Specialty Triage</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
  body{font-family:'Sarabun',sans-serif;max-width:920px;margin:0 auto;padding:32px 24px;color:#1a1a2e;line-height:1.8}
  h1{color:#1a3a6c;font-size:2em;border-bottom:3px solid #1a3a6c;padding-bottom:12px}
  h2{color:#1a3a6c;font-size:1.4em;margin-top:40px;border-left:5px solid #4fc3f7;padding-left:12px}
  h3{color:#0d47a1;font-size:1.15em;margin-top:20px}
  .stats{display:flex;gap:16px;margin:20px 0}
  .stat-box{background:#e3f2fd;border-radius:12px;padding:16px 24px;text-align:center;flex:1}
  .stat-num{font-size:2.2em;font-weight:700;color:#1565c0}
  .stat-label{font-size:.9em;color:#555}
  table{width:100%;border-collapse:collapse;margin:16px 0}
  th{background:#1a3a6c;color:#fff;padding:10px 14px;text-align:left}
  td{padding:10px 14px;border:1px solid #ddd}
  tr:nth-child(even){background:#f5f9ff}
  .step{background:#f8f9fa;border-left:4px solid #4fc3f7;padding:12px 16px;margin:10px 0;border-radius:0 8px 8px 0}
  .info-box{background:#e8f4fd;border-left:5px solid #2196f3;padding:14px 16px;border-radius:0 8px 8px 0;margin:14px 0}
  .warning-box{background:#fff8e1;border-left:5px solid #ffc107;padding:14px 16px;border-radius:0 8px 8px 0;margin:14px 0}
  .error-box{background:#ffebee;border-left:5px solid #f44336;padding:14px 16px;border-radius:0 8px 8px 0;margin:14px 0}
  .mascot-box{background:linear-gradient(135deg,#fff5f5,#ffe4e4);border-left:5px solid #ff8fa3;padding:18px 20px;border-radius:0 12px 12px 0;margin:14px 0}
  .source-item{background:#f0f7ff;padding:10px 16px;margin:8px 0;border-radius:8px}
  .footer{margin-top:48px;padding-top:16px;border-top:1px solid #ddd;font-size:.85em;color:#888;text-align:center}
  @media print{h2{page-break-before:always}.stats{page-break-before:avoid}}
</style>
</head>
<body>
<h1>📖 คู่มือการใช้งาน</h1>
<p><strong>Symptom-to-Specialty Triage</strong> · DADS5001 Final Project</p>
<div class="stats">
  <div class="stat-box"><div class="stat-num">48</div><div class="stat-label">โรคในฐานข้อมูล</div></div>
  <div class="stat-box"><div class="stat-num">132</div><div class="stat-label">อาการที่รองรับ</div></div>
  <div class="stat-box"><div class="stat-num">13</div><div class="stat-label">สาขาแพทย์</div></div>
</div>

<h2>🏥 ส่วนที่ 1 · ภาพรวม App</h2>
<p><strong>Symptom-to-Specialty Triage</strong> ช่วยคนไทยประเมินอาการเบื้องต้นก่อนตัดสินใจไปโรงพยาบาล โดยแนะนำ<strong>สาขาแพทย์ที่เหมาะสม</strong>และ<strong>ระดับความเร่งด่วน</strong> (1–5 ตามมาตรฐาน ED Triage กระทรวงสาธารณสุข)</p>
<div class="error-box">⚠️ <strong>แอปนี้ไม่วินิจฉัยโรค และไม่แทนที่การพบแพทย์จริง</strong></div>
<h3>การไหลของข้อมูล</h3>
<p>🤒 มีอาการ → เลือก Mode → 📊 วิเคราะห์ → 🎯 สาขาแพทย์ + ระดับเร่งด่วน → 💊 ยา &amp; 🏥 รพ. → 🩺 พบแพทย์</p>
<h3>เลือก Mode ไหนดี?</h3>
<table>
<tr><th></th><th>📋 Non-AI Mode</th><th>🤖 AI Mode (น้องอุ่นใน)</th></tr>
<tr><td>วิธีระบุอาการ</td><td>tick checkbox จากรายการ</td><td>พิมพ์หรือเลือกเป็นภาษาไทย</td></tr>
<tr><td>เหมาะกับ</td><td>รู้ว่าตัวเองมีอาการอะไรแน่ชัด</td><td>อยากเล่าอาการแบบธรรมชาติ</td></tr>
<tr><td>ความเร็ว</td><td>เร็วมาก (ไม่ใช้ AI)</td><td>ช้ากว่าเล็กน้อย (เรียก LLM)</td></tr>
<tr><td>Internet</td><td>ไม่ต้องการ</td><td>ต้องการเชื่อมต่อ</td></tr>
<tr><td>Rate limit</td><td>ไม่มี</td><td>20 ครั้ง/session</td></tr>
<tr><td>การวิเคราะห์</td><td>TF-IDF + Bayes scoring</td><td>LLM extract + TF-IDF scoring</td></tr>
</table>

<h2>📋 ส่วนที่ 2 · Non-AI Mode</h2>
<div class="step">1️⃣ เปิดหน้า <strong>Non-AI Mode</strong> จาก sidebar ด้านซ้าย</div>
<div class="step">2️⃣ <strong>เลือกอาการ</strong> จาก checkbox ที่จัดกลุ่มตาม body system — ใช้ช่องค้นหาพิมพ์ชื่ออาการภาษาไทย · เลือกได้หลายอาการพร้อมกัน · ยิ่งเลือกครบ ผลลัพธ์ยิ่งแม่นยำ</div>
<div class="step">3️⃣ <strong>เลือกจังหวัด</strong> (ไม่บังคับ) เพื่อกรองโรงพยาบาลในพื้นที่ที่สะดวก</div>
<div class="step">4️⃣ กดปุ่ม <strong>🔍 วิเคราะห์อาการ</strong></div>
<div class="step">5️⃣ <strong>อ่านผลลัพธ์:</strong> Confidence badge (สูง 🟢 / กลาง 🟡 / ต่ำ 🟠 / ต่ำมาก 🔴) · Top-3 โรค · สาขาแพทย์ · ระดับเร่งด่วน 1–5 · Expander ยา + รพ.</div>
<div class="info-box">💡 ถ้า confidence badge แสดง 🔴 ต่ำมาก ลองเพิ่มอาการ หรือใช้ AI Mode ที่น้องอุ่นในจะถามเพิ่มให้เอง</div>

<h2>🤖 ส่วนที่ 3 · AI Mode — น้องอุ่นใน</h2>
<div class="mascot-box"><strong>🌸 น้องอุ่นใน (Aoon-nai) คือใคร?</strong><br>AI ผู้ช่วยที่มีน้ำเสียงอบอุ่น เรียกผู้ใช้ว่า "พี่" และจะถามคำถามแบบ adaptive เพื่อเก็บข้อมูลอาการให้ครบก่อนวิเคราะห์</div>
<div class="step">1️⃣ เปิดหน้า <strong>AI Mode</strong> จาก sidebar ด้านซ้าย</div>
<div class="step">2️⃣ <strong>เลือก Chief Complaint</strong> จาก 6 หมวด: เป็นไข้ · ไข้หวัด · ปวดหัว · ปวดท้อง · ปวดกล้ามเนื้อ · ผื่น (หรือพิมพ์เองในช่อง free text)</div>
<div class="step">3️⃣ <strong>ตอบคำถามน้องอุ่นใน</strong> แบบ adaptive สูงสุด ~6 ข้อ เช่น ระยะเวลา · ความรุนแรง · อาการร่วม · ปัจจัยเสี่ยง</div>
<div class="step">4️⃣ <strong>อ่านผลลัพธ์:</strong> คำบรรยายจากน้องอุ่นใน + reasoning · Confidence badge · Top-3 โรค + สาขาแพทย์ · red-flag warnings</div>
<div class="step">5️⃣ เลือกจังหวัดกรอง รพ. และดู Expander ยา</div>
<div class="warning-box">⚠️ <strong>Rate limit:</strong> 20 ครั้งต่อ session — รีเฟรชหน้าเพื่อเริ่ม session ใหม่</div>

<h2>⚠️ ส่วนที่ 4 · ข้อจำกัดและคำเตือนด้านการแพทย์</h2>
<div class="error-box">🚨 <strong>อาการฉุกเฉิน — โทร 1669 ทันที หรือไปห้องฉุกเฉินโดยตรง</strong><br>เช่น หัวใจวาย · หายใจไม่ออก · ไม่รู้สึกตัว · เจ็บหน้าอกรุนแรง · อัมพาต/พูดไม่ชัดฉับพลัน (FAST)</div>
<ul>
<li>🔴 <strong>ไม่วินิจฉัยโรค</strong> — ผลลัพธ์คือการแนะนำสาขาแพทย์เท่านั้น ไม่ใช่การวินิจฉัยจากแพทย์จริง</li>
<li>🔴 <strong>ไม่แทนแพทย์</strong> — ควรพบแพทย์เพื่อรับการวินิจฉัยที่ถูกต้อง</li>
<li>🟡 <strong>Confidence ต่ำมาก</strong> — แนะนำให้พบแพทย์โดยตรง ไม่ควรตัดสินใจจากผลนี้เพียงอย่างเดียว</li>
<li>🟡 <strong>ยาในระบบ</strong> — เป็นข้อมูลอ้างอิงทั่วไป ไม่ใช่ใบสั่งยา ห้ามซื้อยาเองโดยไม่ปรึกษาเภสัชกร/แพทย์</li>
<li>🟡 <strong>โรคหายาก/ซับซ้อน</strong> — ฐานข้อมูล 48 โรคครอบคลุมโรคพบบ่อย โรคหายากอาจไม่อยู่ในระบบ</li>
<li>🟢 <strong>ความเป็นส่วนตัว</strong> — log บันทึกแบบ anonymized ไม่มีข้อมูลส่วนตัว</li>
</ul>

<h2>📚 ส่วนที่ 5 · แหล่งข้อมูล</h2>
<div class="source-item">🗃️ <strong>itachi9604/disease-symptom-description-dataset</strong> (Kaggle) — 48 โรค × 132 อาการ (binary matrix)</div>
<div class="source-item">🏥 <strong>ICD-10-TM</strong> (กรมการแพทย์ กระทรวงสาธารณสุข) — Thai Modification ของ International Classification of Diseases</div>
<div class="source-item">🚑 <strong>ED Triage MOPH</strong> — แนวทาง 5-level Triage มาตรฐานห้องฉุกเฉินไทย</div>
<div class="source-item">📊 <strong>data.go.th</strong> — สถิติสาธารณสุขไทย 7 ชุดข้อมูล พ.ศ. 2559–2567</div>
<div class="source-item">☁️ <strong>Snowflake</strong> (AWS ap-southeast-1) — training data, drug mapping, hospital list</div>
<div class="source-item">🍃 <strong>MongoDB Atlas</strong> (AWS ap-southeast-1, M0 Free) — symptom dictionary, disease prevalence, AI session log</div>

<div class="footer">คู่มือฉบับนี้จัดทำเพื่อประกอบโครงการ DADS5001 Final Project · Educational use only<br>หากมีปัญหาการใช้งาน ดูรายละเอียดที่ DISCLAIMER.md ใน GitHub repository</div>
</body>
</html>"""

st.download_button(
    label="📥 ดาวน์โหลดคู่มือ (HTML → เปิดในเบราว์เซอร์ → Ctrl+P → Save as PDF)",
    data=_HTML_MANUAL.encode("utf-8"),
    file_name="คู่มือการใช้งาน_Symptom-to-Specialty-Triage.html",
    mime="text/html",
)
