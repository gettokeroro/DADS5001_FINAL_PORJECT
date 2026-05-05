# Project Memory — DADS5001 Final
> Memory file สำหรับทีม (Kade + พี่เก็ตโตะ) · update เมื่อมีงานใหม่ · ไฟล์นี้ commit ขึ้น git ได้

อัปเดตล่าสุด: **2026-05-05 (Kade)**

---

## 🎯 Project scope ปัจจุบัน

- **Repo หลัก (deployed)**: `gettokeroro/DADS5001_FINAL_PORJECT` (พี่เก็ตโตะเป็นเจ้าของ)
- **Kade fork**: `nuttakarnsinpho610-ai/DADS5001_FINAL_PORJECT` (Kade enhanced version — รอ clone)
- **Tech stack**: Streamlit Cloud · DuckDB · Gemini Flash (น้องอุ่นใน) · MongoDB Atlas (planned)

## ✅ สถานะ Phase

| Phase | สถานะ | หมายเหตุ |
|---|---|---|
| 1 — Population prevalence weighting | ✅ done | data/processed/disease_prevalence.csv |
| 1.5 — Interactive co-symptom follow-up | ✅ done | scoring.py + AI Mode page |
| 2 — Confidence + UI warnings | ✅ done | high/med/low badges |
| 3 — Casual Thai dict expansion | ⬜ ค้าง | สำหรับสัปดาห์หน้า |
| 4 — Add diseases (Influenza, Electrolyte ฯลฯ) | ⬜ ค้าง | |
| 5 — Honest fallback responses | ⬜ ค้าง | |
| 6 — Drug + Hospital data | 🟡 in progress | อ่านด้านล่าง |
| 7 — UI styling sync | 🟡 in progress | batch 1 done (2_AI, 3_Compare) · batch 2 ค้าง 3 หน้า |

## 📦 Phase 6 — Drug + Hospital integration

### ✅ เสร็จแล้ว (2026-05-03 by Kade)

**ไฟล์ใหม่**: `data/processed/disease_drug_mapping_v2_ed.csv`
- 82 ยา × 41 โรค ครอบคลุม itachi disease list ครบ
- Cross-referenced กับ บัญชียาหลักแห่งชาติ 2013 (831 ยา) เพื่อใส่ ED category จริง
- ED breakdown: **ก=53 · ข=6 · ค=4 · ง=7 · จ2=1 · non-ED=11**

**Schema (11 columns) — ใหม่กว่า skeleton เก่า:**

| Column | ตัวอย่าง |
|---|---|
| disease_en | Tuberculosis |
| disease_th | วัณโรค |
| drug_generic | Isoniazid + Rifampicin + Pyrazinamide + Ethambutol |
| drug_th | HRZE (สูตร 6 เดือน) |
| ed_category | ก / ข / ค / ง / จ2 / non-ED |
| ed_category_meaning | "ยาจำเป็นพื้นฐาน — เบิกได้ทุกสิทธิ (🟢)" |
| reimbursement_note | "บัตรทอง/ประกันสังคม/ข้าราชการ เบิกได้ทุกกรณี" |
| indication_th | "วัณโรคปอด initial phase 2 เดือน" |
| dose_note | "1 ครั้ง/วันก่อนอาหาร · ห้ามขาดยา" |
| prescription_tier | otc / pharmacy / doctor / specialist / emergency |
| source_xls_row | 295 (อ้างอิง row ใน บัญชียาหลัก XLS) |

> ⚠ **Breaking change vs Phase 6 skeleton เก่า** (`disease_drug_mapping.csv` 7 cols) — เลยใส่ชื่อใหม่ `_v2_ed` เพื่อไม่ทับของเก่า
> `data_loader.py` ตอนนี้ยังอ่านของเก่าได้ปกติ ไม่พัง · ต้อง refactor เป็น schema ใหม่ภายหลัง

### 🔧 ยังเหลือ (TODO)

1. **Update `utils/data_loader.py`** — ให้อ่าน `disease_drug_mapping_v2_ed.csv` แทน · column mapping ใหม่:
   - `nle_status` → `ed_category`
   - `dosage_note` → `dose_note`
   - `disclaimer_level` → `prescription_tier`
   - + ฟิลด์ใหม่: `ed_category_meaning`, `reimbursement_note`, `prescription_tier`

2. **Update Non-AI + AI Mode pages** — render ใหม่:
   - Badge สีตาม ED (🟢🟡🔵🟣🔴⚪)
   - Banner สีแดง "เฉพาะห้องฉุกเฉิน" สำหรับ tier=emergency
   - Reimbursement note ใน card ใหญ่ๆ ตอบ "เบิกได้ไหม?"

3. **Hospital province filter** ✅ Demo ทำเสร็จ (2026-05-03 by Kade)
   - **Cleaned data**: `data/processed/hospitals_thailand.csv` (UTF-8, 1,581 รพ. × 77 จังหวัด, 9 columns)
     - Columns: province, hospital_th, hospital_en, h_code, affiliation, hospital_type, beds, specialty_note, health_region
     - แปลงจาก source `Project/Data/ข้อมูลทั่วไปของโรงพยาบาล/ha_aod_001-2.csv` (TIS-620 → UTF-8)
   - **Demo standalone**: `hospital_province_demo.py` (root ของ folder) — รันด้วย `streamlit run hospital_province_demo.py`
   - **UI Pattern**: ใช้ `st.selectbox(index=None, placeholder=...)` — built-in autocomplete · พิมพ์ "เชี" → filter เห็น เชียงใหม่/เชียงราย
   - Filter รอง 2 ตัว: ชนิด รพ. (รพช./รพศ./เอกชน 27 ประเภท) + สังกัด
   - Card sort by เตียง descending · แสดงชื่อ + H Code + เขตสุขภาพ + ความเชี่ยวชาญ
   - **TODO ต่อไป**: Integrate เข้า Non-AI + AI Mode pages (รอ Kade ทดสอบ demo ก่อน)
     - Option (ก) sidebar (sticky ทุกหน้า) | (ข) หลัง Top-3 โรค ⭐ แนะนำ | (ค) ทั้งสองที่

4. **MongoDB Atlas ingestion** (Kade's value-add ตามแผน)

## 🎨 Phase 7 — UI / Styling sync

### ✅ Batch 1 · CSS injection across pages (2026-05-05 by Kade)

**ปัญหาที่เจอ:** มีแค่ `app.py` + `pages/1_Non_AI_Mode.py` ที่เรียก `inject_global_css()` · 5 หน้าอื่นข้าม → user สลับหน้าแล้ว navy theme + BG image + sidebar gradient หายหมด กลับเป็น default Streamlit

**Batch 1 — แก้แล้ว:**
- ✅ `pages/2_AI_Mode.py` — เพิ่ม `from utils.styling import inject_global_css` + `inject_global_css()` หลัง set_page_config
- ✅ `pages/3_Compare.py` — เพิ่มเหมือนกัน

**Batch 2 — ยังเหลือ (TODO):**
- ⬜ `pages/4_Disease_Landscape.py`
- ⬜ `pages/5_Cost_Estimator.py`
- ⬜ `pages/6_About.py`

**Palette ปัจจุบัน (ไม่เปลี่ยน, จาก `styling.py` + `config.toml`):**
- accent `#22B7E9` (cyan) · navy text `#02497E`
- BG = `BG_R1.png` + 78% white overlay
- Sidebar = 5-stop gradient (pale → cyan-mid → blue-deep)

## 🗂 ไฟล์ data ที่อยู่ใน OneDrive

- `data/processed/disease_drug_mapping.csv` — Phase 6 skeleton เก่า (11 โรค) **ห้ามลบ** ยังใช้กับ data_loader อยู่
- `data/processed/disease_drug_mapping_v2_ed.csv` — **ใหม่ ⭐** (82 ยา × 41 โรค + ED) **← ใช้ตัวนี้ต่อไป**
- `data/processed/hospitals_thailand.csv` — **ใหม่ ⭐** (1,581 รพ. × 77 จังหวัด UTF-8) **← สำหรับ province filter**
- `data/processed/disease_prevalence.csv` — Phase 1
- `data/processed/disease_specialty_mapping.csv` — Task 1 base
- `data/processed/disease_symptom_long.csv` — itachi long form
- `data/processed/specialty_hospital_hint.csv` — Phase 6 skeleton (12 specialty)
- `data/processed/symptom_dictionary_th.csv` — Task 3 base
- `data/processed/symptom_specificity.csv` — Task 4

## 🧪 ไฟล์ demo / standalone scripts (root)
- `hospital_province_demo.py` — **ใหม่** Streamlit demo ของ province autocomplete pattern (รัน `streamlit run hospital_province_demo.py` เพื่อทดสอบก่อน integrate)

## ⚠ Notes สำคัญ

- **อย่าลบ `disease_drug_mapping.csv`** (เก่า) จนกว่า data_loader.py จะเปลี่ยนเป็น v2 schema แล้ว — ไม่งั้น app crash
- **`cloned old chat.txt`** ใน folder = ประวัติแชต private — **อย่า commit ขึ้น git** (ใส่ใน .gitignore แล้วยิ่งดี)
- **`data/raw/datago/`** = raw datasets ใหญ่ + ลิขสิทธิ์ data.go.th — **อย่า commit** (gitignore เช่นกัน)
- **API key ของ Gemini** ใน `.streamlit/secrets.toml` — gitignored อยู่แล้ว ปลอดภัย
