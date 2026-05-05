"""
AI engine — Gemini Flash wrapper for the 2-step pipeline:

    Thai free-text  →  [STEP 1: extract symptoms]  →  list of symptom_en codes
    list of codes   →  Non-AI scoring (DuckDB SQL)  →  ranked diseases
    ranked diseases →  [STEP 2: narrate result]    →  natural-language Thai answer

Uses structured output (Pydantic) for STEP 1 to guarantee valid JSON.
STEP 2 returns Markdown text for direct render.

Configuration:
    GOOGLE_API_KEY in .streamlit/secrets.toml (local) or
    Streamlit Cloud → Settings → Secrets.

Public API:
    extract_symptoms(text, dictionary, api_key) -> ExtractedSymptoms
    narrate_result(text, ranked_df, mapping_df, api_key) -> str
    full_pipeline(text, dictionary, scoring_arts, mapping_df, api_key) -> AIResult
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# Lazy import to avoid hard dependency if user never opens AI page
def _get_genai():
    import google.generativeai as genai
    return genai


# ---------------------------------------------------------------------------
# Model resolution — Google deprecates model names regularly
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, str] = {}

# Try in order — first working model wins. Newest first.
_FLASH_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
]


def _resolve_model_name(api_key: str) -> str:
    """Find a working Gemini Flash model name from current account.
    Caches result by api_key. Robust against Google's model deprecations."""
    if api_key in _MODEL_CACHE:
        return _MODEL_CACHE[api_key]

    genai = _get_genai()
    genai.configure(api_key=api_key)

    try:
        models = list(genai.list_models())
    except Exception as e:
        # If listing fails, return first candidate and hope for the best
        _MODEL_CACHE[api_key] = _FLASH_CANDIDATES[0]
        return _FLASH_CANDIDATES[0]

    # Build set of names that support generateContent
    available = set()
    for m in models:
        name = m.name.rsplit("/", 1)[-1]
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            available.add(name)

    # Match preferred order
    for c in _FLASH_CANDIDATES:
        if c in available:
            _MODEL_CACHE[api_key] = c
            return c

    # Fallback: any model with "flash" in name (skip experimental)
    for name in sorted(available):
        if "flash" in name.lower() and "exp" not in name.lower():
            _MODEL_CACHE[api_key] = name
            return name

    # Absolute last resort: any model
    if available:
        name = sorted(available)[0]
        _MODEL_CACHE[api_key] = name
        return name

    raise ValueError(
        "No Gemini models available for this API key · "
        "ตรวจ key + permissions ที่ aistudio.google.com"
    )


def list_available_models(api_key: str) -> list[str]:
    """For debugging — list all models accessible with this key."""
    genai = _get_genai()
    genai.configure(api_key=api_key)
    try:
        return sorted([
            m.name.rsplit("/", 1)[-1] for m in genai.list_models()
            if "generateContent" in (getattr(m, "supported_generation_methods", []) or [])
        ])
    except Exception as e:
        return [f"<error: {e}>"]


# ---------------------------------------------------------------------------
# Pydantic schemas (structured output for STEP 1)
# ---------------------------------------------------------------------------

class ExtractedSymptom(BaseModel):
    """One symptom extracted from user free-text."""
    symptom_en: str = Field(description="Symptom code (snake_case English) from dictionary")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0,
                              description="How confident the AI is this symptom matches user's description (0-1)")
    user_phrase: Optional[str] = Field(default=None,
                                       description="The Thai phrase from user that maps to this symptom")


class ExtractedSymptoms(BaseModel):
    """Full extraction result — list of symptoms + duration + extra notes."""
    symptoms: list[ExtractedSymptom] = Field(default_factory=list)
    duration_days: Optional[int] = Field(default=None, description="Duration of symptoms in days, if mentioned")
    notes: Optional[str] = Field(default=None,
                                 description="Any other relevant info (age, gender, history) not in symptom list")


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class AIResult:
    """Full AI pipeline result for the UI to render."""
    extracted: ExtractedSymptoms
    ranked_df: pd.DataFrame                  # Top-K from scoring engine
    narration: str                            # Markdown text from STEP 2
    extract_time_ms: float = 0.0
    narrate_time_ms: float = 0.0
    scoring_time_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# STEP 1: Extract symptoms from free-text
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """You are a medical assistant helping match Thai patient descriptions
to symptom codes from a dictionary.

You will be given:
1. A patient's free-text description in Thai
2. A list of available symptom codes with Thai translations

Your task: extract which symptoms the patient mentions, using ONLY codes from the list.

Rules:
- Output ONLY valid JSON matching the schema below — no markdown, no commentary
- If a Thai phrase isn't in the dictionary, ignore it (don't invent codes)
- Be conservative: only include symptoms clearly stated, not inferred
- For ambiguous phrases, pick the most likely match with lower confidence
- Confidence is 0-1: 1.0 = exact match, 0.5 = inferred, 0.3 = guess

JSON schema:
{
  "symptoms": [
    {"symptom_en": "<code>", "confidence": <float>, "user_phrase": "<original Thai>"},
    ...
  ],
  "duration_days": <int or null>,
  "notes": "<extra context like age, history, or null>"
}
"""


def _build_dictionary_table(dictionary_df: pd.DataFrame, max_items: int = 150) -> str:
    """Format the symptom dictionary as a compact table for the LLM."""
    df = dictionary_df.head(max_items)
    lines = ["symptom_en | symptom_th | symptom_th_alt"]
    for _, row in df.iterrows():
        en = row["symptom_en"]
        th = row.get("symptom_th", "")
        alt = row.get("symptom_th_alt", "") or ""
        lines.append(f"{en} | {th} | {alt}")
    return "\n".join(lines)


def extract_symptoms(
    user_text: str,
    dictionary_df: pd.DataFrame,
    api_key: str,
    model: Optional[str] = None,
) -> tuple[ExtractedSymptoms, float]:
    """
    STEP 1: Use Gemini to extract structured symptom list from Thai free-text.
    Returns (ExtractedSymptoms, elapsed_ms).
    Raises ValueError if API call fails or output is unparseable.
    If model is None, auto-resolves to a working Gemini Flash variant.
    """
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

    if model is None:
        model = _resolve_model_name(api_key)

    dict_table = _build_dictionary_table(dictionary_df)
    prompt = (
        f"{_EXTRACT_SYSTEM_PROMPT}\n\n"
        f"Available symptoms:\n{dict_table}\n\n"
        f"Patient says (Thai):\n```\n{user_text}\n```\n\n"
        "Output (JSON only):"
    )

    m = genai.GenerativeModel(model)
    try:
        response = m.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 2048,
            },
        )
        raw = response.text.strip()
    except Exception as e:
        raise ValueError(f"Gemini API error during extract: {e}") from e

    # Parse + validate
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {raw[:200]!r}") from e

    try:
        result = ExtractedSymptoms(**data)
    except ValidationError as e:
        raise ValueError(f"Gemini output failed schema validation: {e}") from e

    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ---------------------------------------------------------------------------
# STEP 2: Narrate the ranked diseases as Thai paragraphs
# ---------------------------------------------------------------------------

_NARRATE_SYSTEM_PROMPT = """คุณคือ "น้องอุ่นใน (Aoon-nai)" — ผู้ช่วยค้นหาอาการ-โรคที่เป็นมิตร
พูดคุยกับผู้ใช้ภาษาไทยอย่างอบอุ่น แต่ยังคงความเป็นมืออาชีพ

═══ บุคลิกของน้องอุ่นใน ═══
- เรียกตัวเองว่า "น้องอุ่นใน" หรือ "น้อง"
- เรียกผู้ใช้ว่า "พี่" หรือ "คุณ" (เลือกตามบริบท)
- ใช้คำลงท้าย "ครับ/ค่ะ" สลับไปได้ (default = "ครับ")
- ภาษาเข้าใจง่าย ไม่ใช้ศัพท์แพทย์เกินจำเป็น · ถ้าจำเป็นต้องใช้ ให้แปลในวงเล็บ
- น้ำเสียง: ห่วงใย แต่ไม่ตื่นตระหนก · ให้ความมั่นใจแต่ไม่โอ้อวด

═══ ข้อมูลที่จะได้รับ ═══
1. ข้อความต้นฉบับจาก user (ภาษาไทย)
2. ผลวิเคราะห์ Top-3 จาก rule-based engine (TF-IDF) — มี:
   • ชื่อโรค (ภาษาไทย)
   • แผนกแนะนำ
   • ระดับเร่งด่วน 1-5 (1 = ฉุกเฉินทันที, 5 = ไม่เร่งด่วน)
   • อาการเตือนที่ควรระวัง (red flags)

═══ หน้าที่ ═══
เขียนคำตอบ Markdown ภาษาไทย ~5-8 ย่อหน้า ตามโครงนี้:

**ย่อหน้า 1 — ทักทาย + สรุปสิ่งที่ได้ยินจาก user**
"สวัสดีครับพี่ น้องอุ่นในเองนะครับ จากที่พี่เล่ามาว่า [สรุปอาการสั้นๆ ด้วยภาษาง่าย]
น้องเข้าใจว่าน่าเป็นห่วง..."

**ย่อหน้า 2 — อธิบายผล Top-3**
"น้องลองวิเคราะห์เทียบกับฐานข้อมูลที่มี พบว่าอาการของพี่ใกล้เคียงกับ 3 ภาวะนี้นะครับ:
1. **[โรคที่ 1]** — [คำอธิบายภาษาคนทั่วไป 1 ประโยค + เหตุผลที่ตรง]
2. **[โรคที่ 2]** — ...
3. **[โรคที่ 3]** — ..."

**ย่อหน้า 3 — แนะนำแผนก / ระดับเร่งด่วน**
"น้องแนะนำให้พี่ไปพบ **[แผนก]** ก่อนนะครับ
ระดับความเร่งด่วน: **[ระดับ + ความหมาย]**
- ถ้าระดับ 1-2 → ไป ER ทันที
- ถ้าระดับ 3 → นัดวันถัดไปได้
- ถ้าระดับ 4-5 → นัดตามคิวปกติ"

**ย่อหน้า 4 — Red flags ที่ต้องเฝ้าระวัง**
"⚠ ขอเตือนพี่นะครับ — ถ้ามีอาการเหล่านี้ ให้รีบเข้า ER ทันที:
- [red flags จากข้อมูล]"

**ย่อหน้า 5 — ปิดท้าย + disclaimer**
"น้องอุ่นในเป็นแค่ผู้ช่วยตัวเล็กๆ ที่ใช้ฐานข้อมูลค้นหาอาการนะครับ
**ไม่สามารถวินิจฉัยโรคแทนแพทย์จริงได้** — กรุณาปรึกษาแพทย์เพื่อความแน่ใจเสมอ
หวังว่าพี่จะหายไวๆ นะครับ 💙"

**ย่อหน้า 6 — แหล่งอ้างอิง (สำคัญ — ใส่ทุกครั้ง)**
"📚 แหล่งข้อมูลที่น้องใช้:
- itachi9604 disease-symptom dataset (Kaggle, 41 โรค × 132 อาการ)
- แนวทาง ED Triage 5-level ของกระทรวงสาธารณสุข
- ICD-10-TM (กรมการแพทย์)"

═══ ข้อห้าม (สำคัญที่สุด) ═══
❌ ห้ามใช้คำว่า "วินิจฉัย" "แน่นอน" "100%" "ต้องเป็น" "ผลตรวจ"
❌ ห้ามแนะนำยา · dosage · brand
❌ ห้ามแนะนำการรักษาเฉพาะ (เช่น ผ่าตัด, ฉีดยา, ทำหัตถการ)
❌ ห้ามทำราวกับเป็นแพทย์ — น้องอุ่นในเป็นแค่ผู้ช่วยค้นหา ไม่ใช่หมอ
❌ ห้ามตอบสั้น 1-2 ประโยคแบบรีบ — ผู้ใช้ห่วงตัวเอง ต้องอบอุ่น
"""


def narrate_result(
    user_text: str,
    ranked_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    api_key: str,
    model: Optional[str] = None,
    confidence: Optional[dict] = None,
) -> tuple[str, float]:
    """
    STEP 2: Use Gemini to write a Thai narrative explaining the Top-3 ranked diseases.
    Returns (markdown_text, elapsed_ms).
    """
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

    if model is None:
        model = _resolve_model_name(api_key)

    # Join top-K with mapping to get specialty + urgency + red flags
    top_k = ranked_df.head(3)
    enriched = top_k.merge(
        mapping_df, left_on="disease", right_on="disease_en", how="left"
    )

    rows = []
    for _, r in enriched.iterrows():
        rows.append(
            f"- โรค: {r.get('disease_th', r['disease'])}\n"
            f"  แผนกหลัก: {r.get('primary_specialty', '—')}\n"
            f"  Urgency: {int(r['urgency_level']) if pd.notna(r.get('urgency_level')) else '-'} (1=emergent, 5=non-urgent)\n"
            f"  Red flags: {r.get('red_flags', '-') or '-'}"
        )
    ranked_block = "\n".join(rows)

    # Add confidence guidance to prompt if low/medium/very_low
    confidence_block = ""
    if confidence:
        level = confidence.get("level", "medium")
        if level == "very_low":
            # Phase 5: ตอบสั้นๆ · ห้ามวิเคราะห์ Top-3 · ขอข้อมูลเพิ่มอย่างเดียว
            confidence_block = (
                "\n## ⚠ Confidence: VERY LOW (Honest Fallback Mode)\n"
                f"ระบบ rule-based บอกว่า: {confidence.get('reason', '')}\n"
                "**สำคัญมาก — บังคับ:**\n"
                "- ตอบสั้น **ไม่เกิน 3-4 บรรทัด** เท่านั้น\n"
                "- บอก user ตรงๆ ว่าน้องยังให้คำตอบไม่ได้ ข้อมูลอาการน้อยเกินไป\n"
                "- ขอให้ user พิมพ์อาการเพิ่ม: **เวลาที่เป็น · ความรุนแรง · ตำแหน่ง · อาการร่วม**\n"
                "- **ห้าม** วิเคราะห์โรคใน Top-3 · **ห้าม** บอกชื่อโรค · **ห้าม** แนะนำแผนก\n"
                "- เน้น 'ขอข้อมูลเพิ่ม' เป็นหลัก · น้ำเสียงอบอุ่น เรียก user ว่า 'พี่' ตามปกติ\n"
            )
        elif level == "low":
            confidence_block = (
                "\n## ⚠ Confidence: LOW\n"
                f"ระบบ rule-based บอกว่า: {confidence.get('reason', '')}\n"
                "**สำคัญ:** เปิดด้วยการบอก user ตรงๆ ว่าน้องไม่ค่อยมั่นใจ · "
                "อาการที่บอกมายังกว้างเกินไป · "
                "ขอให้ user บอกอาการเพิ่มเติม (เวลาที่เป็น, ความรุนแรง, อาการอื่นๆ) "
                "ก่อนสรุปโรค · อย่าทำราวกับมั่นใจมาก\n"
            )
        elif level == "medium":
            confidence_block = (
                "\n## Confidence: MEDIUM\n"
                f"ระบบ rule-based บอกว่า: {confidence.get('reason', '')}\n"
                "บอก user ว่าระบบเชื่อพอประมาณ · ผลที่ให้ใช้เป็น guide ได้ "
                "แต่ยังไม่ชัด 100% · ให้ปรึกษาแพทย์เพื่อยืนยัน\n"
            )
        else:  # high
            confidence_block = (
                "\n## Confidence: HIGH\n"
                "ระบบ rule-based มั่นใจในผล · ตอบได้แบบปกติ\n"
            )

    prompt = (
        f"{_NARRATE_SYSTEM_PROMPT}\n"
        f"{confidence_block}"
        f"## User text\n{user_text}\n\n"
        f"## Top-3 result\n{ranked_block}\n\n"
        "## Answer (Thai Markdown):\n"
    )

    m = genai.GenerativeModel(model)
    try:
        response = m.generate_content(
            prompt,
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 2048,
            },
        )
        text = response.text.strip()
    except Exception as e:
        raise ValueError(f"Gemini API error during narrate: {e}") from e

    elapsed = (time.perf_counter() - t0) * 1000
    return text, elapsed


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_pipeline(
    user_text: str,
    dictionary_df: pd.DataFrame,
    scoring_arts,
    mapping_df: pd.DataFrame,
    api_key: str,
    method: str = "tfidf",
    top_k: int = 3,
) -> AIResult:
    """Run STEP 1 -> scoring -> STEP 2."""
    from utils.scoring import predict

    try:
        extracted, t_ext = extract_symptoms(user_text, dictionary_df, api_key)


        if not extracted.symptoms:
            return AIResult(
                extracted=extracted,
                ranked_df=pd.DataFrame(),
                narration=(
                    "สวัสดีครับพี่ น้องอุ่นในเองนะครับ - "
                    "ตอนนี้น้องยังไม่สามารถจับคู่อาการที่พี่บอกมา "
                    "กับฐานข้อมูลที่มีได้ ลองพิมพ์อาการให้ชัดขึ้นนะครับ "
                    "(เช่น 'ไอแห้ง 3 วัน + เจ็บคอ') หรือสลับไปใช้ "
                    "Non-AI Mode เพื่อติ๊ก checkbox ได้โดยตรง"
                ),
                extract_time_ms=t_ext,
            )

        symptom_codes = [s.symptom_en for s in extracted.symptoms]
        ts = time.perf_counter()
        ranked = predict(symptom_codes, scoring_arts, method=method, top_k=top_k)
        t_score = (time.perf_counter() - ts) * 1000

        narration, t_narr = narrate_result(user_text, ranked, mapping_df, api_key)

        return AIResult(
            extracted=extracted,
            ranked_df=ranked,
            narration=narration,
            extract_time_ms=t_ext,
            narrate_time_ms=t_narr,
            scoring_time_ms=t_score,
        )

    except ValueError as e:
        return AIResult(
            extracted=ExtractedSymptoms(),
            ranked_df=pd.DataFrame(),
            narration="",
            error=str(e),
        )


def check_rate_limit(session_state, max_calls: int = 20) -> tuple[bool, int]:
    counter_key = "ai_call_counter"
    n = session_state.get(counter_key, 0)
    if n >= max_calls:
        return False, n
    session_state[counter_key] = n + 1
    return True, n + 1


def reset_rate_limit(session_state):
    session_state["ai_call_counter"] = 0
