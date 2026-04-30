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
    model: str = "gemini-1.5-flash",
) -> tuple[ExtractedSymptoms, float]:
    """
    STEP 1: Use Gemini to extract structured symptom list from Thai free-text.
    Returns (ExtractedSymptoms, elapsed_ms).
    Raises ValueError if API call fails or output is unparseable.
    """
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

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

_NARRATE_SYSTEM_PROMPT = """คุณเป็นผู้ช่วยทางการแพทย์ที่อธิบายผลการวิเคราะห์อาการให้ผู้ป่วยภาษาไทย

จะได้รับ:
1. ข้อความต้นฉบับจากผู้ป่วย
2. ผลการวิเคราะห์ Top-3 จากระบบ rule-based (TF-IDF) — มี: โรค, แผนกแนะนำ, ระดับเร่งด่วน, อาการเตือน

หน้าที่ของคุณ: เขียนคำอธิบาย Markdown สั้นๆ ภาษาไทย ความยาว ~3-5 ย่อหน้า ที่:

1. **สรุปสั้น** ว่าเข้าได้กับโรคใดบ้าง (3 อันดับแรก) พร้อมเหตุผลย่อ
2. **แนะนำแผนก/ความเร่งด่วน** ที่ควรไปก่อน — ใช้ภาษาคนทั่วไป ไม่ใช้ศัพท์แพทย์เกินความจำเป็น
3. **อาการที่ต้องระวัง** (red flags) — สั้น ตรงประเด็น
4. **ปิดท้ายด้วย disclaimer** "ไม่ใช่คำวินิจฉัย · ปรึกษาแพทย์จริงเสมอ"

ห้าม: ระบุว่าผู้ป่วยเป็นโรคอะไรแน่นอน · แนะนำยา · บอกการรักษา · ทำราวกับเป็นแพทย์
"""


def narrate_result(
    user_text: str,
    ranked_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> tuple[str, float]:
    """
    STEP 2: Use Gemini to write a Thai narrative explaining the Top-3 ranked diseases.
    Returns (markdown_text, elapsed_ms).
    """
    t0 = time.perf_counter()
    genai = _get_genai()
    genai.configure(api_key=api_key)

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
            f"  ระดับเร่งด่วน: {int(r['urgency_level']) if pd.notna(r.get('urgency_level')) else '—'} (1=ฉุกเฉิน, 5=ไม่เร่งด่วน)\n"
            f"  อาการเตือน: {r.get('red_flags', '—') or '—'}"
        )
    ranked_block = "\n".join(rows)

    prompt = (
        f"{_NARRATE_SYSTEM_PROMPT}\n\n"
        f"## ข้อความผู้ป่วย\n{user_text}\n\n"
        f"## ผลวิเคราะห์ Top-3\n{ranked_block}\n\n"
        "## คำอธิบาย (Markdown):\n"
    )

    m = genai.GenerativeModel(model)
    try:
        response = m.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 1024,
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
    scoring_arts,           # ScoringArtifacts from utils.scoring
    mapping_df: pd.DataFrame,
    api_key: str,
    method: str = "tfidf",
    top_k: int = 3,
) -> AIResult:
    """
    Run STEP 1 → scoring → STEP 2 sequentially.
    Returns AIResult with all intermediate results + timings.
    On error, returns AIResult with .error set (UI shows graceful fallback).
    """
    from utils.scoring import predict

    try:
        # STEP 1
        extracted, t_ext = extract_symptoms(user_text, dictionary_df, api_key)

        # No symptoms found → bail out gracefully
        if not extracted.symptoms:
            return AIResult(
                extracted=extracted,
                ranked_df=pd.DataFrame(),
                narration="⚠ ไม่สามารถระบุอาการที่ตรงกับฐานข้อมูลได้จากข้อความที่ให้มา · "
                          "ลองพิมพ์อาการให้ชัดเจนขึ้น (เช่น \"ไอแห้ง 3 วัน + เจ็บคอ\") "
                          "หรือสลับไปใช้ Non-AI Mode เพื่อติ๊ก checkbox โดยตรง",
                extract_time_ms=t_ext,
            )

        # Score with Non-AI engine
        symptom_codes = [s.symptom_en for s in extracted.symptoms]
        ts = time.perf_counter()
        ranked = predict(symptom_codes, scoring_arts, method=method, top_k=top_k)
        t_score = (time.perf_counter() - ts) * 1000

        # STEP 2
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


# ---------------------------------------------------------------------------
# Rate limiter (session-based)
# ---------------------------------------------------------------------------

def check_rate_limit(session_state, max_calls: int = 20) -> tuple[bool, int]:
    """Check + increment rate limit. Returns (allowed, calls_so_far)."""
    counter_key = "ai_call_counter"
    n = session_state.get(counter_key, 0)
    if n >= max_calls:
        return False, n
    session_state[counter_key] = n + 1
    return True, n + 1


def reset_rate_limit(session_state):
    """Reset call counter (debug/admin)."""
    session_state["ai_call_counter"] = 0
