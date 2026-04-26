# 🩺 Symptom-to-Specialty Triage

> DADS5001 Final Project · Data-centric Streamlit app with AI add-on

ตัวช่วยตัดสินใจว่าอาการของคุณควรปรึกษาแพทย์แผนกไหน · พร้อมเปรียบเทียบโหมด **Non-AI** (rule-based, TF-IDF) กับ **AI** (LLM + RAG)

⚠ **Educational use only · ไม่ใช่คำวินิจฉัยทางการแพทย์**

---

## ปัญหาที่แก้

คนไทยมักกูเกิ้ลอาการตัวเอง → เจอข้อมูลปลายทางที่ไม่ตรงบริบท → ตื่นตระหนกหรือรักษาผิด → ไม่ไปหาหมอจริง หรือไปผิดแผนก

แอปนี้แนะนำ **แผนก/ประเภทแพทย์** ที่ควรไป + **ระดับความเร่งด่วน 1–5** จากอาการที่ระบุ

---

## Features

| # | Page | Mode | Status |
|---|---|---|---|
| 1 | 🩺 Non-AI Mode | Checkbox + TF-IDF/Bayes scoring | ✅ Live |
| 2 | 🤖 AI Mode | Free-text + LLM + RAG | 🚧 Phase 4 |
| 3 | ⚖️ Compare | Side-by-side Non-AI vs AI | 🚧 Phase 5 |
| 4 | 📊 Disease Landscape | Thai stats from data.go.th | 🚧 Phase 5 |
| 5 | 💰 Cost Estimator | ค่ารักษาประมาณการ (Feature B) | 🚧 Phase 5 |
| 6 | ℹ️ About | Methodology + data sources | ✅ Live |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/gettokeroro/DADS5001_FINAL_PORJECT.git
cd DADS5001_FINAL_PORJECT

# 2. Install
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 3. (optional) Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml — add your LLM API key when ready

# 4. Run
streamlit run app.py
```

App จะเปิดที่ <http://localhost:8501>

---

## Project structure

```
DADS5001_FINAL_PORJECT/
├── app.py                          ← Home page
├── pages/                          ← Streamlit multi-pages
│   ├── 1_Non_AI_Mode.py
│   ├── 2_AI_Mode.py
│   ├── 3_Compare.py
│   ├── 4_Disease_Landscape.py
│   ├── 5_Cost_Estimator.py
│   └── 6_About.py
├── utils/
│   ├── scoring.py                  ← TF-IDF + Bayes scoring engine
│   └── data_loader.py              ← Streamlit-cached loaders
├── data/
│   ├── raw/                        ← itachi_train.csv, itachi_test.csv
│   └── processed/                  ← mapping, dictionary, specificity
├── notebooks/
│   ├── eda_specificity.ipynb       ← EDA + viz + eval comparison
│   └── figures/                    ← Generated PNGs
├── .streamlit/
│   ├── config.toml                 ← Theme + page config
│   └── secrets.toml.example        ← Template for LLM keys
├── requirements.txt
├── runtime.txt                     ← Python 3.11
├── README.md                       ← (this file)
├── DISCLAIMER.md                   ← Medical disclaimer ฉบับเต็ม
└── GIT_SETUP.md                    ← วิธี push ขึ้น GitHub
```

---

## Methodology

### Non-AI mode (Phase 3 done)

- **Engine:** TF-IDF scoring + Naive Bayes (เปรียบเทียบ)
- **Input:** checkbox 121 อาการ (ภาษาไทย+อังกฤษย่อ) จัดกลุ่มตาม body system
- **Pipeline:**
  1. User ติ๊ก symptom checkboxes
  2. Compute TF-IDF score per disease: `score = Σ freq(s|d) × idf(s) × coverage_bonus`
  3. Map disease → ICD-10 → primary/secondary specialty + urgency 1-5
  4. Display top-3 + red-flag warnings
- **Eval:** Top-1 hit = 100%, Top-3 hit = 100% (10 cases · `utils.scoring.DEFAULT_EVAL_CASES`)

### AI mode (Phase 4 — under construction)

- **Pipeline:** Thai free-text → LLM extract structured symptoms → match dataset → narrate result + reasoning
- **Grounding:** RAG จาก disease description + ICD-10-TM
- **Guardrail:** input filter, JSON schema (Pydantic) validation, rate limit per session, fallback to Non-AI mode

---

## Data sources

### Symptom-Disease backbone

- [itachi9604 (Kaggle)](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) — 4,920 rows · 41 diseases × 132 symptoms binary

### Thai standards

- ICD-10-TM Ontology (กรมการแพทย์ MOPH)
- ED Triage MOPH 5-level guidelines

### Thai disease statistics ([data.go.th](https://www.data.go.th))

- Top OPD/IPD ranking 2564
- 20 ลำดับโรคสูงสุด IPD 298 กลุ่มโรค 2559–2566 (time series)
- โรคติดต่อ/เฝ้าระวัง 2567
- ประชากรตามอายุ 2565
- ข้อมูลทั่วไปของโรงพยาบาล 2566

### Constructed by team

- `data/processed/disease_specialty_mapping.csv` — 41 diseases × ICD-10 chapter × specialty × urgency × red flags
- `data/processed/symptom_dictionary_th.csv` — 132 symptoms × Thai translation × body system
- `data/processed/symptom_specificity.csv` — 132 symptoms × IDF score

---

## Mandatory requirements (DADS5001)

- ✅ **Streamlit multi-pages** — 7 pages (`app.py` + 6 in `pages/`)
- ✅ **DuckDB + pandas** — `utils/data_loader.py:get_duckdb_connection()`
- 🚧 **External database** — Supabase (Phase 5)
- ✅ **2 modes (Non-AI vs AI)** — separate pages + Compare page
- ✅ **Streamlit cache 3 types** —
  `@st.cache_data` (DataFrames),
  `@st.cache_resource` (DB connection, scoring artifacts),
  `st.session_state` (selected symptoms, language, history)
- 🚧 **Slide presentation** — 15 min, 8 sections
- 🚧 **GitHub submission** — this repo

---

## Tech stack

- **Frontend / App:** Streamlit
- **Analytics:** DuckDB · pandas · numpy · scikit-learn
- **Visualization:** Plotly · matplotlib · seaborn
- **AI / LLM:** TBD (Gemini / Claude / Gemma)
- **External DB:** Supabase (planned)
- **Hosting:** Streamlit Community Cloud

---

## Team

| Member | Role |
|---|---|
| _TBD_ | Data Engineer · ดูแล data ingestion + DuckDB |
| _TBD_ | Non-AI Engineer · scoring + rule engine |
| _TBD_ | AI Engineer · LLM pipeline + RAG |
| _TBD_ | UI / Visualization · Streamlit + Plotly |
| _TBD_ | PM / QA · slide + rehearsal |

---

## License

Educational use only · DADS5001 final project · See [`DISCLAIMER.md`](DISCLAIMER.md) for full disclaimer
