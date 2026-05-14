# AI Mode Redesign — Architecture & Design Decisions

> **Status:** Mockup approved (v4) · waiting for code refactor of `pages/2_AI_Mode.py`
> **Date:** 2026-05-10
> **Owner:** Kade (design + mockup) → Getto (refactor)
> **Mockup:** `mockups/ai_mode_redesign_v4.html` (เปิด browser ดูได้เลย)

---

## 🎯 Vision

**As-is:** User พิมพ์ context ยาวๆ ใน text box → AI ตีความ → วินิจฉัย (UX แบบ chatbot)

**To-be:** Question-based UX → User คลิกตัวเลือกตาม decision tree → กด "ขอคำวินิจฉัย" → หมอ narrate Top-3 โรค + ยา + รพ. (UX แบบ patient interview)

---

## 🧭 5 Architectural Decisions

| # | Topic | Choice | Rationale |
|---|---|---|---|
| 1 | Question source | **D — Hybrid** | Symptom tree (data-driven จาก `disease_symptom_long.csv`) + Gemma narrate ตอนจบ → scale ได้ + ภาษาธรรมชาติ |
| 2 | Q1 type | **B — Chief complaints** (6 items) | คนทั่วไปคิดเป็นอาการ ไม่ใช่ body system · natural สุดสำหรับ first-time user |
| 3 | Tree depth | **Adaptive + soft cap 5-6** | จบเมื่อ shortlist <3 โรค หรือ user กด "ขอคำวินิจฉัย" anytime |
| 4 | Free text logic | **Match → continue · Mismatch → ถามซ้ำ/ตัวเลือกอื่น** | ใช้ `str.contains` กับ `symptom_dictionary_th.csv` (Phase 3 batch 2 = 600+ tokens) |
| 5 | Diagnosis page | **Doctor narrate + drug + hospital** | Gemma เล่าเรื่องสไตล์หมอใจดี · ข้อมูล structured คล้าย Non-AI mode |

---

## 📋 Q1 Chief Complaints (6 + free text)

1. **เป็นไข้** → high_fever, mild_fever, chills
2. **ไข้หวัด / ภูมิแพ้** → cough, runny_nose, continuous_sneezing, throat_irritation
3. **ปวดหัว** → headache, dizziness
4. **ปวดท้อง / ทางเดินอาหาร** → stomach_pain, nausea, vomiting, diarrhoea, constipation
5. **ปวดกล้ามเนื้อ / ข้อ** → joint_pain, muscle_pain, back_pain, fatigue
6. **ผื่น / ผิวหนัง** → itching, skin_rash, red_spots_over_body
7. **อื่นๆ พิมพ์เอง** → free text → match กับ symptom_dictionary_th

---

## 🌳 Tree Branching (demo path: ปวดท้อง)

```
Q1: วันนี้คุณเป็นอะไรมาคะ?
└─ ปวดท้อง
   └─ Q2: ปวดท้องลักษณะไหนคะ?
      ├─ แสบยอดอก / กรดไหลย้อน  → acidity
      ├─ ปวดท้องน้อย              → belly_pain
      ├─ ปวดเสียดท้อง / จุก       → indigestion
      ├─ ท้องเสีย                  → diarrhoea
      └─ อื่นๆ พิมพ์เอง           → free text screen
         └─ Q3: มีอาการอื่นๆด้วยไหมคะ?
            ├─ คลื่นไส้ / อ้วก     → nausea, vomiting
            ├─ เรอเปรี้ยว           → acidity sub-symptom
            ├─ ถ่ายดำ / ถ่ายเป็นเลือด → bloody_stool
            ├─ ไม่มีอาการอื่น     → skip
            └─ อื่นๆ พิมพ์เอง    → free text screen
               └─ [ขอคำวินิจฉัย] → diagnosis page
```

**State storage:** `st.session_state["picked_symptoms"]` = list of `symptom_en`

---

## 🩺 Diagnosis Page Layout

**Left column** (main content):
- Header bar: "คำวินิจฉัยโรคของคุณคือ :"
- 3 narrative cards (Gemma-generated · style: kindly doctor explaining)
  - Top-1: full narrative + drug card + hospital card
  - Top-2, Top-3: shorter narrative
- Each disease shows: confidence %, Thai name, English name

**Right column** (200-240px):
- Doctor mascot character (cream-yellow body, glasses, stethoscope)
- "คุณหมอใจดี" + tagline

**Drug card:** ยาแนะนำ + ED tier badge (ก/ข/ค/ง color-coded)
**Hospital card:** รพ. แนะนำ + จำนวนเตียง (จาก `hospitals_thailand.csv`)

---

## 🎨 Palette & Visual

**Pastel palette:**
- Question box: white bg + `#85B7EB` border (Blue 200)
- Stepper: `#B5D4F4` active (Blue 100)
- Diag button: `#C0DD97` (Green 100/200) — คลิกเด่น
- Text primary: `#042C53` (Navy 900) — high contrast
- Text secondary: `#185FA5` (Navy 600)

**Character (kawaii medical mascot — original SVG):**
- Nurse: pink round body + white nurse cap with red cross + heart on belly + simple smile
- Doctor: cream-yellow body + glasses + stethoscope + lab coat hint
- ทั้ง 2 ตัวเป็น original SVG ใน mockup HTML — production สามารถ swap เป็นรูป AI-gen หรือ illustration จริงได้ทีหลัง

**Hover state on options:** light green pastel `#EAF3DE` (Green 50) → คลิก `#C0DD97`

---

## 🔧 Implementation Plan (next session)

### Files ที่ต้องแก้
- **`pages/2_AI_Mode.py`** — รื้อ + refactor ใหม่ (เปลี่ยน UX จาก textarea → button-based)
- **`utils/ai_engine.py`** — ปรับ Gemma prompt: รับ symptom list + return structured Top-3 narrative
- **`utils/symptom_tree.py`** (new) — define question tree + branching logic จาก `disease_symptom_long.csv`
- **`utils/styling.py`** — อาจเพิ่ม CSS injection สำหรับ AI Mode (pastel palette + mascot styling)

### Components ที่ reuse จาก Non-AI
- `scoring.py` — Top-3 disease ranking (consistent กับ Non-AI ✅)
- `data_loader.py` — load symptom_dictionary_th + disease_drug_mapping_v2_ed + hospitals
- `symptom_dictionary_th.csv` — Phase 3 batch 2 expand แล้ว (121/121 user-facing 100%)

### State management
- `st.session_state["picked_symptoms"]` = list[str] (symptom_en)
- `st.session_state["screen"]` = "Q1" | "Q2" | "Q3" | "freetext" | "diagnosis"
- `st.session_state["history"]` = list (สำหรับปุ่ม "กลับ")

---

## 📝 Iteration History

- **v1** — initial layout · navy heavy palette · placeholder character box
- **v2** — pastel palette · darker fonts · custom kawaii mascot SVG (round pink body + nurse cap)
- **v3** — hover state เปลี่ยนเป็น green pastel · มาสคอตขนาดใหญ่ขึ้น · "คุณป้าใจดี" → "คุณพยาบาลใจดี" · ลบ + บนหน้าออก เปลี่ยนเป็นหัวใจที่หน้าท้อง
- **v4** ⭐ — fix input bg ขาว · เพิ่ม "ดำเนินการต่อ" button หลัง match · ขยาย mock dictionary 24→80 tokens

---

## 🚦 Open questions (รอ Getto + Kade ตกลงก่อน refactor)

1. **State persistence** ระหว่างหน้า: ใช้ `st.session_state` ตามปกติพอ หรือต้อง URL-based routing?
2. **Mobile responsive:** mockup ออกแบบ desktop-first (1fr 240px grid) — mobile อาจต้อง stack vertical
3. **Character images final:** ใช้ SVG ใน mockup ต่อเลย หรือจะ AI-gen รูปจริง? (ขนาดควร 200×200 PNG/SVG)
4. **AI engine prompt** สำหรับ narrative: ใช้ Gemma Flash + few-shot 2-3 ตัวอย่าง · tone "หมอใจดี ภาษาเข้าใจง่าย" · max 2-3 ประโยคต่อโรค
