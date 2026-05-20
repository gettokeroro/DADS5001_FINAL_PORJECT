"""
Page 5 · คู่มือการใช้งาน / แหล่งข้อมูล — Symptom-to-Specialty Triage
"""
import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import init_session_state, render_disclaimer_sidebar

st.set_page_config(page_title="คู่มือ / แหล่งข้อมูล", page_icon="📖", layout="wide")
init_session_state()
render_disclaimer_sidebar()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("คู่มือการใช้งาน")
st.caption("Symptom-to-Specialty Triage · DADS5001 Final Project")

_c1, _c2, _c3 = st.columns(3)
_c1.metric("โรคในฐานข้อมูล", "48 โรค")
_c2.metric("อาการที่รองรับ", "132 อาการ")
_c3.metric("สาขาแพทย์", "13 สาขา")

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 1 · ภาพรวม
# ─────────────────────────────────────────────────────────────
st.markdown("## ส่วนที่ 1 · ภาพรวม")
st.markdown(
    "**Symptom-to-Specialty Triage** ช่วยคนไทยประเมินอาการเบื้องต้นก่อนตัดสินใจไปโรงพยาบาล "
    "โดยแนะนำ **สาขาแพทย์ที่เหมาะสม** และ **ระดับความเร่งด่วน** "
    "(1–5 ตามมาตรฐาน ED Triage กระทรวงสาธารณสุข)\n\n"
    "> แอปนี้ไม่วินิจฉัยโรค และไม่แทนที่การพบแพทย์จริง"
)

_mermaid_html = (
    '<div style="text-align:center;padding:8px;">' +
    '<div class="mermaid">flowchart LR\n' +
    '    A["มีอาการ"] --> B{"เลือก Mode"}\n' +
    '    B --> C["Non-AI Mode"]\n' +
    '    B --> D["AI Mode (น้องอุ่นใน)"]\n' +
    '    C --> E["เลือก checkbox\\n132 อาการ"]\n' +
    '    D --> F["พิมพ์/เลือก\\nอาการภาษาไทย"]\n' +
    '    E --> G["วิเคราะห์"]\n' +
    '    F --> G\n' +
    '    G --> H["สาขาแพทย์\\n+ ระดับเร่งด่วน"]\n' +
    '    H --> I["ยา & โรงพยาบาล"]\n' +
    '    I --> J["พบแพทย์"]\n' +
    '</div></div>' +
    '<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>' +
    '<script>mermaid.initialize({startOnLoad:true,theme:"neutral",flowchart:{curve:"basis"}});</script>'
)
components.html(_mermaid_html, height=230)

st.markdown("### เลือก Mode ไหนดี?")
st.markdown(
    "| | Non-AI Mode | AI Mode (น้องอุ่นใน) |\n"
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
st.markdown("## ส่วนที่ 2 · Non-AI Mode")
st.info("เหมาะสำหรับผู้ที่รู้ชัดว่ามีอาการอะไร และต้องการผลรวดเร็วโดยไม่ใช้ AI")

_steps_nonai = [
    ("1.", "เปิดหน้า **Non-AI Mode** จาก sidebar ด้านซ้าย"),
    (
        "2.",
        "**เลือกอาการ** จาก checkbox ที่จัดกลุ่มตาม body system "
        "— ใช้ช่องค้นหาพิมพ์ชื่ออาการภาษาไทยเพื่อกรอง "
        "· เลือกได้หลายอาการพร้อมกัน · ยิ่งเลือกครบ ผลลัพธ์ยิ่งแม่นยำ",
    ),
    ("3.", "**เลือกจังหวัด** (ไม่บังคับ) เพื่อกรองโรงพยาบาลในพื้นที่ที่สะดวก"),
    ("4.", "กดปุ่ม **วิเคราะห์อาการ** เพื่อรับผลลัพธ์"),
    (
        "5.",
        "**อ่านผลลัพธ์:** Confidence badge (สูง / กลาง / ต่ำ / ต่ำมาก) "
        "· Top-3 โรคที่น่าจะเป็น · สาขาแพทย์แนะนำ · ระดับเร่งด่วน 1–5 "
        "· ดูยาและโรงพยาบาลใน expander",
    ),
]
for _num, _text in _steps_nonai:
    st.markdown(f"**{_num}** {_text}")

st.info(
    "เคล็ดลับ: ถ้า confidence แสดงระดับ 'ต่ำมาก' "
    "ลองเพิ่มอาการ หรือใช้ AI Mode ที่น้องอุ่นในจะถามเพิ่มให้เอง"
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 3 · AI Mode
# ─────────────────────────────────────────────────────────────
st.markdown("## ส่วนที่ 3 · AI Mode — น้องอุ่นใน")

# mascot card — single-line HTML (per Streamlit markdown HTML rule)
st.markdown('<div style="background:#fdf6f6;border-radius:12px;padding:18px 22px;border-left:4px solid #e8a0a0;margin-bottom:16px;"><strong>น้องอุ่นใน (Aoon-nai)</strong><p style="margin:6px 0 0 0;color:#555;font-size:0.95em;">AI ผู้ช่วยที่มีน้ำเสียงอบอุ่น เรียกผู้ใช้ว่า "พี่" และจะถามคำถามเพิ่มเติมแบบ adaptive เพื่อเก็บข้อมูลอาการให้ครบก่อนส่งผลวิเคราะห์</p></div>', unsafe_allow_html=True)

st.info("เหมาะสำหรับผู้ที่อยากเล่าอาการแบบภาษาธรรมชาติ หรือยังไม่แน่ใจว่ามีอาการอะไรบ้าง")

_steps_ai = [
    ("1.", "เปิดหน้า **AI Mode** จาก sidebar ด้านซ้าย"),
    (
        "2.",
        "**เลือก Chief Complaint** (อาการหลัก) จาก 6 หมวด: "
        "เป็นไข้ · ไข้หวัด · ปวดหัว · ปวดท้อง · ปวดกล้ามเนื้อ · ผื่น "
        "(หรือพิมพ์อาการอื่นในช่อง free text)",
    ),
    (
        "3.",
        "**ตอบคำถามน้องอุ่นใน** — ถามแบบ adaptive สูงสุด ~6 ข้อ "
        "เช่น ระยะเวลา · ความรุนแรง · อาการร่วม · ปัจจัยเสี่ยง",
    ),
    (
        "4.",
        "**อ่านผลลัพธ์:** คำบรรยายจากน้องอุ่นในพร้อม reasoning "
        "· Confidence badge · Top-3 โรค + สาขาแพทย์แนะนำ "
        "· แจ้งเตือน red-flag symptoms ถ้ามี",
    ),
    ("5.", "**เลือกจังหวัด** เพื่อกรองโรงพยาบาล และดูยาใน expander"),
]
for _num, _text in _steps_ai:
    st.markdown(f"**{_num}** {_text}")

st.warning(
    "Rate limit: AI Mode จำกัด **20 ครั้งต่อ session** เพื่อควบคุมค่าใช้จ่าย API "
    "หากเกินขีดจำกัด ให้รีเฟรชหน้าเพื่อเริ่ม session ใหม่"
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 4 · ข้อจำกัดและคำเตือน
# ─────────────────────────────────────────────────────────────
st.markdown("## ส่วนที่ 4 · ข้อจำกัดและคำเตือนด้านการแพทย์")

st.error(
    "🚨 อาการฉุกเฉิน — โทร 1669 ทันที หรือไปห้องฉุกเฉินโดยตรง\n\n"
    "เช่น หัวใจวาย · หายใจไม่ออก · ไม่รู้สึกตัว · เจ็บหน้าอกรุนแรง · อัมพาต/พูดไม่ชัดฉับพลัน"
)

_warnings = [
    ("🔴", "**ไม่วินิจฉัยโรค** — ผลลัพธ์คือ _การแนะนำสาขาแพทย์_ เท่านั้น ไม่ใช่การวินิจฉัยจากแพทย์จริง"),
    ("🔴", "**ไม่แทนแพทย์** — ข้อมูลที่ได้เป็นเพียงการประเมินเบื้องต้น ควรพบแพทย์เพื่อรับการวินิจฉัยที่ถูกต้อง"),
    (
        "🟡",
        "**Confidence ต่ำมาก** — หากผลลัพธ์แสดง confidence ต่ำมาก "
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
st.markdown("## ส่วนที่ 5 · แหล่งข้อมูล")

with st.expander("Symptom-Disease Backbone Dataset"):
    st.markdown(
        "- **[itachi9604/disease-symptom-description-dataset]"
        "(https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)** (Kaggle)\n"
        "  - 48 โรค × 132 อาการ (binary matrix) หลัง augment เพิ่ม 7 โรคพบบ่อยในไทย\n"
        "  - รองรับ TF-IDF scoring + Naive Bayes classifier"
    )

with st.expander("มาตรฐานการแพทย์ไทย"):
    st.markdown(
        "- **ICD-10-TM** — Thai Modification ของ International Classification of Diseases "
        "(กรมการแพทย์ กระทรวงสาธารณสุข)\n"
        "- **ED Triage MOPH** — แนวทาง 5-level Triage มาตรฐานห้องฉุกเฉินไทย\n"
        "- **disease_specialty_mapping.csv** — สร้างเอง: mapping 48 โรค × 13 สาขาแพทย์"
    )

with st.expander("สถิติสาธารณสุขไทย (data.go.th)"):
    st.markdown(
        "- การจัดอันดับโรคของผู้ป่วยนอก/ใน พ.ศ. 2564\n"
        "- 20 ลำดับโรคสูงสุด IPD 298 กลุ่มโรค พ.ศ. 2559–2566 (time series 8 ปี)\n"
        "- ผู้ป่วยนอก 21 กลุ่มโรค พ.ศ. 2559–2565\n"
        "- การเข้ารับบริการ OPD พ.ศ. 2566\n"
        "- โรคติดต่อ/เฝ้าระวัง พ.ศ. 2567\n"
        "- ประชากรตามอายุ สิทธิหลักประกันสุขภาพ พ.ศ. 2565\n"
        "- ข้อมูลทั่วไปของโรงพยาบาล พ.ศ. 2566"
    )

with st.expander("Cloud Storage"):
    st.markdown(
        "- **Snowflake** (AWS ap-southeast-1) — training data, drug mapping, hospital list\n"
        "- **MongoDB Atlas** (AWS ap-southeast-1, M0 Free) "
        "— symptom dictionary, disease prevalence, AI session log"
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# Download HTML
# ─────────────────────────────────────────────────────────────
st.markdown("## ดาวน์โหลดคู่มือ")
st.markdown(
    "บันทึกคู่มือเป็นไฟล์ HTML → เปิดในเบราว์เซอร์ → **Ctrl+P** → Save as PDF"
)

_HTML_MANUAL = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>คู่มือการใช้งาน — Symptom-to-Specialty Triage</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
  body{font-family:'Sarabun',sans-serif;max-width:900px;margin:0 auto;padding:32px 24px;color:#222;line-height:1.8;font-size:15px}
  h1{font-size:1.9em;font-weight:700;border-bottom:2px solid #222;padding-bottom:10px;margin-bottom:6px}
  h2{font-size:1.3em;font-weight:600;margin-top:40px;padding-top:12px;border-top:1px solid #ddd}
  h3{font-size:1.1em;font-weight:600;margin-top:18px;color:#333}
  .stats{display:flex;gap:16px;margin:20px 0}
  .stat-box{border:1px solid #ddd;border-radius:8px;padding:14px 20px;text-align:center;flex:1}
  .stat-num{font-size:2em;font-weight:700}
  .stat-label{font-size:.85em;color:#666}
  table{width:100%;border-collapse:collapse;margin:14px 0;font-size:.95em}
  th{border-bottom:2px solid #222;padding:8px 12px;text-align:left;font-weight:600}
  td{padding:8px 12px;border-bottom:1px solid #eee}
  .step{padding:10px 0 10px 16px;border-left:3px solid #ccc;margin:8px 0}
  .note{background:#f8f8f8;padding:12px 16px;border-radius:6px;margin:12px 0;font-size:.95em}
  .warn{background:#fffbf0;padding:12px 16px;border-radius:6px;margin:12px 0;border-left:3px solid #e0a800}
  .danger{background:#fff5f5;padding:12px 16px;border-radius:6px;margin:12px 0;border-left:3px solid #c0392b}
  .mascot{background:#fdf6f6;padding:16px 18px;border-radius:8px;border-left:3px solid #e8a0a0;margin:12px 0}
  .source{padding:8px 0;border-bottom:1px solid #eee}
  .footer{margin-top:48px;padding-top:14px;border-top:1px solid #ddd;font-size:.85em;color:#999;text-align:center}
  @media print{h2{page-break-before:always}.stats{page-break-before:avoid}}
</style>
</head>
<body>
<h1>คู่มือการใช้งาน</h1>
<p>Symptom-to-Specialty Triage · DADS5001 Final Project</p>
<div class="stats">
  <div class="stat-box"><div class="stat-num">48</div><div class="stat-label">โรคในฐานข้อมูล</div></div>
  <div class="stat-box"><div class="stat-num">132</div><div class="stat-label">อาการที่รองรับ</div></div>
  <div class="stat-box"><div class="stat-num">13</div><div class="stat-label">สาขาแพทย์</div></div>
</div>

<h2>ส่วนที่ 1 · ภาพรวม</h2>
<p><strong>Symptom-to-Specialty Triage</strong> ช่วยคนไทยประเมินอาการเบื้องต้นก่อนตัดสินใจไปโรงพยาบาล โดยแนะนำ<strong>สาขาแพทย์ที่เหมาะสม</strong>และ<strong>ระดับความเร่งด่วน</strong> (1–5 ตามมาตรฐาน ED Triage กระทรวงสาธารณสุข)</p>
<div class="note">แอปนี้ไม่วินิจฉัยโรค และไม่แทนที่การพบแพทย์จริง</div>
<h3>การไหลของข้อมูล</h3>
<p>มีอาการ → เลือก Mode → วิเคราะห์ → สาขาแพทย์ + ระดับเร่งด่วน → ยา &amp; โรงพยาบาล → พบแพทย์</p>
<h3>เลือก Mode ไหนดี?</h3>
<table>
<tr><th></th><th>Non-AI Mode</th><th>AI Mode (น้องอุ่นใน)</th></tr>
<tr><td>วิธีระบุอาการ</td><td>tick checkbox จากรายการ</td><td>พิมพ์หรือเลือกเป็นภาษาไทย</td></tr>
<tr><td>เหมาะกับ</td><td>รู้ว่าตัวเองมีอาการอะไรแน่ชัด</td><td>อยากเล่าอาการแบบธรรมชาติ</td></tr>
<tr><td>ความเร็ว</td><td>เร็วมาก (ไม่ใช้ AI)</td><td>ช้ากว่าเล็กน้อย (เรียก LLM)</td></tr>
<tr><td>Internet</td><td>ไม่ต้องการ</td><td>ต้องการเชื่อมต่อ</td></tr>
<tr><td>Rate limit</td><td>ไม่มี</td><td>20 ครั้ง/session</td></tr>
<tr><td>การวิเคราะห์</td><td>TF-IDF + Bayes scoring</td><td>LLM extract + TF-IDF scoring</td></tr>
</table>

<h2>ส่วนที่ 2 · Non-AI Mode</h2>
<div class="step"><strong>1.</strong> เปิดหน้า Non-AI Mode จาก sidebar ด้านซ้าย</div>
<div class="step"><strong>2.</strong> เลือกอาการจาก checkbox ที่จัดกลุ่มตาม body system — ใช้ช่องค้นหาพิมพ์ชื่ออาการภาษาไทย · เลือกได้หลายอาการพร้อมกัน · ยิ่งเลือกครบ ผลลัพธ์ยิ่งแม่นยำ</div>
<div class="step"><strong>3.</strong> เลือกจังหวัด (ไม่บังคับ) เพื่อกรองโรงพยาบาลในพื้นที่ที่สะดวก</div>
<div class="step"><strong>4.</strong> กดปุ่มวิเคราะห์อาการ</div>
<div class="step"><strong>5.</strong> อ่านผลลัพธ์: Confidence badge (สูง / กลาง / ต่ำ / ต่ำมาก) · Top-3 โรค · สาขาแพทย์ · ระดับเร่งด่วน 1–5 · expander ยา + รพ.</div>
<div class="note">เคล็ดลับ: ถ้า confidence แสดงระดับ "ต่ำมาก" ลองเพิ่มอาการ หรือใช้ AI Mode</div>

<h2>ส่วนที่ 3 · AI Mode — น้องอุ่นใน</h2>
<div class="mascot"><strong>น้องอุ่นใน (Aoon-nai)</strong><br>AI ผู้ช่วยที่มีน้ำเสียงอบอุ่น เรียกผู้ใช้ว่า "พี่" และจะถามคำถามแบบ adaptive เพื่อเก็บข้อมูลอาการให้ครบก่อนวิเคราะห์</div>
<div class="step"><strong>1.</strong> เปิดหน้า AI Mode จาก sidebar ด้านซ้าย</div>
<div class="step"><strong>2.</strong> เลือก Chief Complaint จาก 6 หมวด: เป็นไข้ · ไข้หวัด · ปวดหัว · ปวดท้อง · ปวดกล้ามเนื้อ · ผื่น (หรือพิมพ์เองในช่อง free text)</div>
<div class="step"><strong>3.</strong> ตอบคำถามน้องอุ่นใน แบบ adaptive สูงสุด ~6 ข้อ</div>
<div class="step"><strong>4.</strong> อ่านผลลัพธ์: คำบรรยายจากน้องอุ่นใน + reasoning · Confidence badge · Top-3 โรค + สาขาแพทย์ · red-flag warnings</div>
<div class="step"><strong>5.</strong> เลือกจังหวัดกรองโรงพยาบาล และดูยาใน expander</div>
<div class="warn">Rate limit: 20 ครั้งต่อ session — รีเฟรชหน้าเพื่อเริ่ม session ใหม่</div>

<h2>ส่วนที่ 4 · ข้อจำกัดและคำเตือนด้านการแพทย์</h2>
<div class="danger">🚨 อาการฉุกเฉิน — โทร 1669 ทันที หรือไปห้องฉุกเฉินโดยตรง<br>เช่น หัวใจวาย · หายใจไม่ออก · ไม่รู้สึกตัว · เจ็บหน้าอกรุนแรง · อัมพาต/พูดไม่ชัดฉับพลัน</div>
<ul>
<li>🔴 <strong>ไม่วินิจฉัยโรค</strong> — ผลลัพธ์คือการแนะนำสาขาแพทย์เท่านั้น</li>
<li>🔴 <strong>ไม่แทนแพทย์</strong> — ควรพบแพทย์เพื่อรับการวินิจฉัยที่ถูกต้อง</li>
<li>🟡 <strong>Confidence ต่ำมาก</strong> — แนะนำให้พบแพทย์โดยตรง</li>
<li>🟡 <strong>ยาในระบบ</strong> — ข้อมูลอ้างอิงทั่วไป ไม่ใช่ใบสั่งยา</li>
<li>🟡 <strong>โรคหายาก/ซับซ้อน</strong> — อาจไม่อยู่ในฐานข้อมูล 48 โรค</li>
<li>🟢 <strong>ความเป็นส่วนตัว</strong> — log บันทึกแบบ anonymized ไม่มีข้อมูลส่วนตัว</li>
</ul>

<h2>ส่วนที่ 5 · แหล่งข้อมูล</h2>
<div class="source"><strong>itachi9604/disease-symptom-description-dataset</strong> (Kaggle) — 48 โรค × 132 อาการ (binary matrix)</div>
<div class="source"><strong>ICD-10-TM</strong> (กรมการแพทย์ กระทรวงสาธารณสุข) — Thai Modification ของ International Classification of Diseases</div>
<div class="source"><strong>ED Triage MOPH</strong> — แนวทาง 5-level Triage มาตรฐานห้องฉุกเฉินไทย</div>
<div class="source"><strong>data.go.th</strong> — สถิติสาธารณสุขไทย 7 ชุดข้อมูล พ.ศ. 2559–2567</div>
<div class="source"><strong>Snowflake</strong> (AWS ap-southeast-1) — training data, drug mapping, hospital list</div>
<div class="source"><strong>MongoDB Atlas</strong> (AWS ap-southeast-1, M0 Free) — symptom dictionary, disease prevalence, AI session log</div>

<div class="footer">DADS5001 Final Project · Educational use only</div>
</body>
</html>"""

st.download_button(
    label="ดาวน์โหลดคู่มือ (HTML)",
    data=_HTML_MANUAL.encode("utf-8"),
    file_name="คู่มือการใช้งาน_Symptom-to-Specialty-Triage.html",
    mime="text/html",
)
