"""
AI engine -- Gemini Flash wrapper for the 2-step pipeline:

    Thai free-text  ->  [STEP 1: extract symptoms]  ->  list of symptom_en codes
    list of codes   ->  Non-AI scoring (DuckDB SQL)  ->  ranked diseases
    ranked diseases ->  [STEP 2: narrate result]    ->  natural-language Thai answer

Public API:
    extract_symptoms(text, dictionary, api_key) -> ExtractedSymptoms
    narrate_result(text, ranked_df, mapping_df, api_key) -> str
    narrate_cards(picked_symptoms_th, ranked_df, mapping_df, api_key) -> list[str]
    full_pipeline(text, dictionary, scoring_arts, mapping_df, api_key) -> AIResult
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


def _get_genai():
    import google.generativeai as genai
    return genai


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, str] = {}

_FLASH_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
]


def _resolve_model_name(api_key: str) -> str:
    if api_key in _MODEL_CACHE:
        return _MODEL_CACHE[api_key]

    genai = _get_genai()
    genai.configure(api_key=api_key)

    try:
        models = list(genai.list_models())
    except Exception:
        _MODEL_CACHE[api_key] = _FLASH_CANDIDATES[0]
        return _FLASH_CANDIDATES[0]

    available = set()
    for m in models:
        name = m.name.rsplit("/", 1)[-1]
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            available.add(name)

    for c in _FLASH_CANDIDATES:
        if c in available:
            _MODEL_CACHE[api_key] = c
            return c

    for name in sorted(available):
        if "flash" in name.lower() and "exp" not in name.lower():
            _MODEL_CACHE[api_key] = name
            return name

    if available:
        name = sorted(available)[0]
        _MODEL_CACHE[api_key] = name
        return name

    raise ValueError("No Gemini models available for this API key")


def list_available_models(api_key: str) -> list[str]:
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
# Pydantic schemas
# ---------------------------------------------------------------------------

class ExtractedSymptom(BaseModel):
    symptom_en: str = Field(description="Symptom code from dictionary")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    user_phrase: Optional[str] = Field(default=None)


class ExtractedSymptoms(BaseModel):
    symptoms: list[ExtractedSymptom] = Field(default_factory=list)
    duration_days: Optional[int] = Field(default=None)
    notes: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class AIResult:
    extracted: ExtractedSymptoms
    ranked_df: pd.DataFrame
    narration: str
    extract_time_ms: float = 0.0
    narrate_time_ms: float = 0.0
    scoring_time_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# STEP 1: Extract symptoms
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """You are a medical assistant helping match Thai patient descriptions
to symptom codes from a dictionary.

Rules:
- Output ONLY valid JSON -- no markdown, no commentary
- Only include symptoms clearly stated, not inferred
- Confidence is 0-1: 1.0 = exact match, 0.5 = inferred, 0.3 = guess

JSON schema:
{
  "symptoms": [{"symptom_en": "<code>", "confidence": <float>, "user_phrase": "<Thai>"}],
  "duration_days": <int or null>,
  "notes": "<extra context or null>"
}
"""


def _build_dictionary_table(dictionary_df: pd.DataFrame, max_items: int = 150) -> str:
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
# STEP 2: Narrate ranked diseases
# ---------------------------------------------------------------------------

_NARRATE_SYSTEM_PROMPT = (
    'คุณคือ "น้องอุ่นใน (Aoon-nai)" -- ผู้ช่วยค้นหาอาการ-โรคที่เป็นมิตร\n'
    "เขียนคำตอบ Markdown ภาษาไทย ~5-8 ย่อหน้า\n"
    "ห้ามใช้คำว่า 'วินิจฉัย' 'แน่นอน' '100%' 'ต้องเป็น'\n"
    "ห้ามแนะนำยา dosage brand\n"
)


def narrate_result(
    user_text: str,
    ranked_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    api_key: str,
    model: Optional[str] = None,
    confidence: Optional[dict] = None,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

    if model is None:
        model = _resolve_model_name(api_key)

    top_k = ranked_df.head(3)
    enriched = top_k.merge(
        mapping_df, left_on="disease", right_on="disease_en", how="left"
    )

    rows = []
    for _, r in enriched.iterrows():
        rows.append(
            f"- โรค: {r.get('disease_th', r['disease'])}\n"
            f"  แผนก: {r.get('primary_specialty', '---')}\n"
            f"  Urgency: {int(r['urgency_level']) if pd.notna(r.get('urgency_level')) else '-'}\n"
            f"  Red flags: {r.get('red_flags', '-') or '-'}"
        )
    ranked_block = "\n".join(rows)

    confidence_block = ""
    if confidence:
        level = confidence.get("level", "medium")
        if level == "very_low":
            confidence_block = (
                "\n## WARNING: VERY LOW confidence\n"
                "ตอบสั้น ไม่เกิน 3-4 บรรทัด ขอข้อมูลเพิ่ม ห้ามบอกชื่อโรค\n"
            )
        elif level == "low":
            confidence_block = (
                "\n## WARNING: LOW confidence\n"
                "บอก user ว่าไม่ค่อยมั่นใจ ขอให้บอกอาการเพิ่มเติม\n"
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
# STEP 2b: Card-style narrations (JSON -> 3 short strings)
# ---------------------------------------------------------------------------

_NARRATE_CARDS_PROMPT = (
    'คุณคือ "คุณหมอใจดี" -- แพทย์ที่อธิบายผลการวิเคราะห์แบบกระชับและอบอุ่น\n'
    "ส่งคืน JSON เท่านั้น รูปแบบ:\n"
    '{"d1": "2-3 ประโยคโรคที่ 1", "d2": "1-2 ประโยคโรคที่ 2", "d3": "1-2 ประโยคโรคที่ 3"}\n'
    "ห้ามใช้คำว่า 'วินิจฉัย' 'แน่นอน' '100%' ห้ามแนะนำยาชื่อเฉพาะ ห้ามตอบเป็น markdown\n"
)


def narrate_cards(
    picked_symptoms_th: list[str],
    ranked_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    api_key: str,
    model: Optional[str] = None,
) -> tuple[list[str], float]:
    """
    Card-style narrations for AI Mode result screen (Phase 8 redesign).
    Returns (list of 3 short Thai narration strings, elapsed_ms).
    """
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

    if model is None:
        model = _resolve_model_name(api_key)

    top_k = ranked_df.head(3)
    enriched = top_k.merge(mapping_df, left_on="disease", right_on="disease_en", how="left")

    disease_lines = []
    for i, (_, r) in enumerate(enriched.iterrows()):
        disease_lines.append(
            f"โรค #{i+1}: {r.get('disease_th', r['disease'])} ({r['disease']})\n"
            f"  Urgency: {int(r['urgency_level']) if pd.notna(r.get('urgency_level')) else 5}\n"
            f"  Red flags: {r.get('red_flags', '') or '-'}"
        )

    symptoms_str = ", ".join(picked_symptoms_th) if picked_symptoms_th else "ไม่ระบุ"

    prompt = (
        f"{_NARRATE_CARDS_PROMPT}\n\n"
        f"## อาการที่ user เลือก\n{symptoms_str}\n\n"
        "## Top-3 โรค\n" + "\n".join(disease_lines) + "\n\n"
        "## JSON Output:"
    )

    m = genai.GenerativeModel(model)
    try:
        response = m.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
                "max_output_tokens": 1024,
            },
        )
        raw = response.text.strip()
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        return ["", "", ""], elapsed

    try:
        data = json.loads(raw)
        narrations = [
            str(data.get("d1", "")),
            str(data.get("d2", "")),
            str(data.get("d3", "")),
        ]
    except (json.JSONDecodeError, AttributeError):
        narrations = ["", "", ""]

    elapsed = (time.perf_counter() - t0) * 1000
    return narrations, elapsed


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
                    "กับฐานข้อมูลที่มีได้ ลองพิมพ์อาการให้ชัดขึ้นนะครับ"
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


def check_rate_limit(
    session_state,
    max_calls: int = 20,
    counter_key: str = "ai_call_counter",
) -> tuple[bool, int]:
    """Check and increment rate-limit counter. Returns (allowed, new_count)."""
    n = session_state.get(counter_key, 0)
    if n >= max_calls:
        return False, n
    session_state[counter_key] = n + 1
    return True, n + 1


def reset_rate_limit(session_state, counter_key: str = "ai_call_counter"):
    session_state[counter_key] = 0
