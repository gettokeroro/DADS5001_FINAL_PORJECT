# 🚀 Git Setup — push skeleton ขึ้น GitHub

> Repo ปลายทาง: <https://github.com/gettokeroro/DADS5001_FINAL_PORJECT>
>
> ตอนนี้ repo บน GitHub มีแค่ `README.md` ของ "Initial commit" — เราจะ push skeleton ทั้งโฟลเดอร์ขึ้นไปทับ

---

## 1️⃣ เปิด terminal ในโฟลเดอร์ project นี้

ที่เครื่องคุณ:

```bash
cd "C:\Users\piyawat.a\OneDrive - National Institute of Development Administration\Claude-Cowork\PROJECTS\ทำ Web app\Data centric app  ที่มีโหมด AI"
```

(หรือเปิด terminal ใน VS Code ที่ folder นี้ก็ได้)

---

## 2️⃣ Init git แล้วเชื่อมกับ remote

```bash
git init
git branch -M main
git remote add origin https://github.com/gettokeroro/DADS5001_FINAL_PORJECT.git
```

---

## 3️⃣ ดึง initial commit จาก GitHub มาก่อน (เพื่อรวมประวัติ)

```bash
git fetch origin
git reset --soft origin/main
```

**คำอธิบาย:** `--soft` ทำให้ไฟล์ทั้งหมดยังอยู่ใน working directory แต่ history ของ remote มาแล้ว · พอ commit ครั้งหน้า history จะต่อกันเรียบ ไม่มี force-push

---

## 4️⃣ Stage + commit ไฟล์ทั้งหมด

```bash
# ดูก่อนว่า .gitignore กรองอะไรออกไหม
git status

# Stage ทุกอย่าง
git add .

# Commit
git commit -m "Initial skeleton — Streamlit multi-page app + Non-AI scoring engine

- 7 pages (Home + 6 in pages/)
- utils/scoring.py (TF-IDF + Bayes, Top-1 100%/Top-3 100% on 10-case eval)
- utils/data_loader.py (3 Streamlit cache types)
- data/processed/: disease_specialty_mapping (41), symptom_dictionary_th (132), specificity (132)
- notebooks/eda_specificity.ipynb + figures/
- requirements.txt, runtime.txt, .gitignore
- README.md, DISCLAIMER.md, GIT_SETUP.md"
```

---

## 5️⃣ Push ขึ้น GitHub

```bash
git push -u origin main
```

ครั้งแรกจะถามรหัสผ่าน — ใช้ **Personal Access Token (PAT)** แทน password
(ดูวิธีสร้าง: <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>)

หรือใช้ **GitHub CLI**:

```bash
gh auth login           # ครั้งแรก ครั้งเดียว
git push -u origin main
```

---

## ✅ ตรวจสอบ

เปิด <https://github.com/gettokeroro/DADS5001_FINAL_PORJECT> — ควรเห็นไฟล์ทั้งหมดถูก upload

ถ้าโอเคแล้ว ทีมคนอื่นทำ:

```bash
git clone https://github.com/gettokeroro/DADS5001_FINAL_PORJECT.git
cd DADS5001_FINAL_PORJECT
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Bonus: Deploy ไป Streamlit Community Cloud

1. ไปที่ <https://share.streamlit.io>
2. Sign in ด้วย GitHub
3. คลิก "New app" → เลือก repo `gettokeroro/DADS5001_FINAL_PORJECT`
4. Branch: `main` · Main file: `app.py` · Python: 3.11
5. Advanced settings → Secrets: paste content จาก `.streamlit/secrets.toml.example` (เปลี่ยนเป็น key จริง)
6. Deploy → ได้ URL `https://<your-app>.streamlit.app`

ใส่ live URL นี้ใน README.md ตอนส่งงาน

---

## 🆘 Troubleshooting

**Q: `git push` แล้ว reject เพราะ "non-fast-forward"?**
A: รัน `git pull origin main --rebase` ก่อน แล้วลอง push ใหม่

**Q: ลืม `.gitignore` แล้ว commit secrets ไปแล้ว?**
A: ลบ secret + commit ใหม่ + **revoke key เก่าทันที** (history ยังเห็นได้แม้ลบไฟล์)

**Q: ขนาด repo ใหญ่เพราะ data?**
A: เพิ่ม `data/raw/*.csv` ใน `.gitignore` (เอา comment ออก) แล้วทีมต้องโหลดเอง · หรือใช้ Git LFS

**Q: Thai filename ใน OneDrive ทำให้ git ช้า?**
A: ลอง clone ไปใส่ folder ปกติบนเครื่อง (ไม่ใช่ใน OneDrive) จะเร็วกว่า
