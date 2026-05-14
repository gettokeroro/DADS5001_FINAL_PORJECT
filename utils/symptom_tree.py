"""
utils/symptom_tree.py — Decision tree logic for AI Mode redesign (Phase 8)

Drives the button-based question flow:
  Q1 (chief complaint) → Q2 (sub-symptom) → Q3+ (co-symptoms) → diagnosis

Reuses existing scoring engine (suggest_co_symptoms, predict) — no AI calls here.
All functions are pure: input → output, no side effects on session_state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# -----------------------------------------------------------------------------
# Q1: Chief complaints — top-level entry options shown on first screen
# -----------------------------------------------------------------------------
# Each entry maps a "chief complaint" bucket to:
#   - label        : Thai display text shown on Q1 button
#   - sublabel     : English symptom codes shown beneath label (for transparency)
#   - symptoms     : symptom_en list ที่จะ pre-suggest ใน Q2
#   - q2_prompt    : คำถาม Q2 ที่ปรับตาม chief complaint
# All symptom_en codes verified to exist in symptom_dictionary_th.csv (2026-05-10)
CHIEF_COMPLAINTS = [
    {
        "id": "fever",
        "label": "เป็นไข้",
        "sublabel": "high_fever, chills",
        "symptoms": ["high_fever", "mild_fever", "chills"],
        "q2_prompt": "เป็นไข้แบบไหนคะ?",
    },
    {
        "id": "cold",
        "label": "ไข้หวัด / ภูมิแพ้",
        "sublabel": "cough, runny_nose, sneezing",
        "symptoms": ["cough", "runny_nose", "continuous_sneezing", "throat_irritation", "phlegm"],
        "q2_prompt": "อาการไหนเด่นคะ?",
    },
    {
        "id": "headache",
        "label": "ปวดหัว",
        "sublabel": "headache, dizziness",
        "symptoms": ["headache", "dizziness", "pain_behind_the_eyes"],
        "q2_prompt": "ปวดหัวลักษณะไหนคะ?",
    },
    {
        "id": "stomach",
        "label": "ปวดท้อง / ทางเดินอาหาร",
        "sublabel": "stomach_pain, nausea, diarrhoea",
        "symptoms": ["stomach_pain", "abdominal_pain", "belly_pain", "acidity",
                     "indigestion", "nausea", "vomiting", "diarrhoea", "constipation"],
        "q2_prompt": "ปวดท้องลักษณะไหนคะ?",
    },
    {
        "id": "musculoskeletal",
        "label": "ปวดกล้ามเนื้อ / ข้อ",
        "sublabel": "joint_pain, muscle_pain, back_pain",
        "symptoms": ["joint_pain", "muscle_pain", "back_pain", "fatigue",
                     "knee_pain", "neck_pain", "cramps"],
        "q2_prompt": "เจ็บตรงไหนคะ?",
    },
    {
        "id": "skin",
        "label": "ผื่น / ผิวหนัง",
        "sublabel": "itching, skin_rash, red_spots",
        "symptoms": ["itching", "skin_rash", "red_spots_over_body",
                     "blister", "pus_filled_pimples"],
        "q2_prompt": "ลักษณะผื่นเป็นยังไงคะ?",
    },
]

CHIEF_BY_ID = {c["id"]: c for c in CHIEF_COMPLAINTS}

# Special sentinel codes
SKIP_CODE = "__skip__"
FREETEXT_CODE = "__freetext__"


# -----------------------------------------------------------------------------
# Output type — shared by all get_*_options() functions
# -----------------------------------------------------------------------------
@dataclass
class TreeOption:
    """A single button option to render in a question screen."""
    symptom_en: str            # code OR sentinel (__skip__, __freetext__)
    label_th: str              # main button label
    sublabel: Optional[str] = None   # small text underneath
    is_skip: bool = False      # True for "ไม่มีอาการอื่น" / freetext branches
    meta: dict = field(default_factory=dict)  # arbitrary extras


# -----------------------------------------------------------------------------
# Q1 — Top-level chief complaints
# -----------------------------------------------------------------------------
def get_q1_options() -> list[dict]:
    """Return raw CHIEF_COMPLAINTS list (caller renders directly)."""
    return [c.copy() for c in CHIEF_COMPLAINTS]


# -----------------------------------------------------------------------------
# Q2 — Sub-symptoms specific to chosen chief complaint
# -----------------------------------------------------------------------------
def get_q2_options(
    chief_complaint_id: str,
    dictionary_df: pd.DataFrame,
) -> list[TreeOption]:
    """
    Build Q2 options from CHIEF_COMPLAINTS[id].symptoms, looking up Thai labels.
    Always appends a "อื่นๆ พิมพ์เอง" freetext option at the end.
    """
    cc = CHIEF_BY_ID.get(chief_complaint_id)
    if cc is None:
        return _freetext_only()

    out: list[TreeOption] = []
    for sym_en in cc["symptoms"]:
        match = dictionary_df[dictionary_df["symptom_en"] == sym_en]
        if match.empty:
            continue
        row = match.iloc[0]
        out.append(TreeOption(
            symptom_en=sym_en,
            label_th=row.get("symptom_th") or sym_en,
            sublabel=sym_en,
        ))
    out.append(_freetext_option())
    return out


# -----------------------------------------------------------------------------
# Q3+ — Adaptive co-symptom suggestions based on what's already picked
# -----------------------------------------------------------------------------
def get_q3plus_options(
    picked_symptoms: list[str],
    arts,  # ScoringArtifacts (from utils.scoring)
    dictionary_df: pd.DataFrame,
    top_k: int = 5,
    exclude: Optional[set[str]] = None,
) -> list[TreeOption]:
    """
    Wrap suggest_co_symptoms() and format for the UI.
    Adds a "ไม่มีอาการอื่น" skip option + "อื่นๆ พิมพ์เอง" freetext option.

    `exclude` — symptom_en codes ที่ไม่ต้องเสนอ (e.g. ที่ user เคยเห็นแล้ว)
    """
    from utils.scoring import suggest_co_symptoms

    if not picked_symptoms:
        return _freetext_only()

    skip_set = set(picked_symptoms) | (exclude or set())
    try:
        co_df = suggest_co_symptoms(picked_symptoms, arts, top_k=top_k * 2)
    except Exception:
        return _freetext_only()

    out: list[TreeOption] = []
    for _, row in co_df.iterrows():
        sym_en = row["symptom"]
        if sym_en in skip_set:
            continue
        match = dictionary_df[dictionary_df["symptom_en"] == sym_en]
        if match.empty:
            continue
        sym_th = match.iloc[0].get("symptom_th") or sym_en
        n_dis = int(row.get("n_diseases_have_it", 0) or 0)
        out.append(TreeOption(
            symptom_en=sym_en,
            label_th=sym_th,
            sublabel=f"พบใน {n_dis} โรคที่ใกล้เคียง",
        ))
        if len(out) >= top_k:
            break

    out.append(TreeOption(
        symptom_en=SKIP_CODE,
        label_th="ไม่มีอาการอื่น",
        sublabel="ขอคำวินิจฉัยเลย",
        is_skip=True,
    ))
    out.append(_freetext_option())
    return out


# -----------------------------------------------------------------------------
# Adaptive stopping rule
# -----------------------------------------------------------------------------
def is_tree_done(
    picked_symptoms: list[str],
    arts,
    questions_asked: int,
    soft_cap: int = 6,
    confidence_threshold: float = 0.6,
    min_symptoms: int = 2,
) -> bool:
    """
    Decide whether to auto-route to diagnosis (true) or ask another question (false).

    Stops when ANY of:
      - questions_asked >= soft_cap (default 6)
      - Top-1 disease primary_score >= confidence_threshold (ranking is clear)

    Always asks at least until len(picked_symptoms) >= min_symptoms.
    """
    if len(picked_symptoms) < min_symptoms:
        return False
    if questions_asked >= soft_cap:
        return True

    from utils.scoring import predict
    try:
        ranked = predict(picked_symptoms, arts, method="tfidf", top_k=3)
        if ranked is None or ranked.empty:
            return False
        top_score = float(ranked.iloc[0].get("primary_score", 0) or 0)
        return top_score >= confidence_threshold
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Free-text matching against symptom_dictionary_th
# -----------------------------------------------------------------------------
def match_freetext(
    text: str,
    dictionary_df: pd.DataFrame,
) -> Optional[TreeOption]:
    """
    Lookup user's typed input against the symptom dictionary.
    Mirrors the str.contains pattern in pages/1_Non_AI_Mode.py (line 134-138):
      checks symptom_th / symptom_th_alt / symptom_en / ui_label.
    Restricts to is_user_facing == True.

    Returns the first matching row as a TreeOption, or None if no match.
    """
    q = (text or "").strip().lower()
    if not q:
        return None

    df = dictionary_df
    if "is_user_facing" in df.columns:
        df = df[df["is_user_facing"] == True]

    cols = [c for c in ["symptom_th", "symptom_th_alt", "symptom_en", "ui_label"]
            if c in df.columns]

    # Two-pass: exact/substring first, then prefix-overlap fallback (Thai word stems)
    _MIN_PFX = 4  # minimum prefix length for Thai stem matching (≈ 1-2 Thai chars)

    prefix_candidate: Optional[TreeOption] = None  # best prefix match (used if no exact)

    for _, row in df.iterrows():
        sym_en = str(row.get("symptom_en") or "")
        label_th = str(row.get("symptom_th") or sym_en)
        for col in cols:
            cell = str(row.get(col) or "").lower()
            if not cell:
                continue
            # 1) Direct substring match
            if q in cell:
                return TreeOption(
                    symptom_en=sym_en,
                    label_th=label_th,
                    sublabel=f"matched in `{col}`",
                    meta={"matched_column": col, "matched_text": q},
                )
            # 2) Comma-separated alts — bidirectional check
            pieces = [p.strip().lower() for p in cell.split(",")] if "," in cell else [cell]
            for piece in pieces:
                if not piece:
                    continue
                if q in piece or piece in q:
                    return TreeOption(
                        symptom_en=sym_en,
                        label_th=label_th,
                        sublabel=f"matched in `{col}` (alt)",
                        meta={"matched_column": col, "matched_alt": piece},
                    )
                # 3) Prefix-overlap (Thai word sharing first stem, e.g. หายใจ-)
                if prefix_candidate is None and len(q) >= _MIN_PFX and len(piece) >= _MIN_PFX:
                    q_pfx = q[:_MIN_PFX]
                    if piece.startswith(q_pfx) or q.startswith(piece[:_MIN_PFX]):
                        prefix_candidate = TreeOption(
                            symptom_en=sym_en,
                            label_th=label_th,
                            sublabel=f"matched in `{col}` (prefix)",
                            meta={"matched_column": col, "matched_prefix": q_pfx},
                        )

    return prefix_candidate  # None if no match at all


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _freetext_option() -> TreeOption:
    return TreeOption(
        symptom_en=FREETEXT_CODE,
        label_th="อื่นๆ พิมพ์เอง",
        sublabel="free text — match กับ symptom dictionary",
        is_skip=True,
    )


def _freetext_only() -> list[TreeOption]:
    return [_freetext_option()]


# -----------------------------------------------------------------------------
# Smoke test (run with: python utils/symptom_tree.py)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=== Q1 options ===")
    for cc in get_q1_options():
        print(f"  [{cc['id']}] {cc['label']} — {len(cc['symptoms'])} symptoms")

    print("\n=== Loading dictionary ===")
    dict_df = pd.read_csv(Path(__file__).parent.parent / "data/processed/symptom_dictionary_th.csv")
    print(f"  loaded {len(dict_df)} rows")

    print("\n=== Q2 options for 'stomach' ===")
    for opt in get_q2_options("stomach", dict_df):
        print(f"  • {opt.label_th:30} ({opt.symptom_en})")

    print("\n=== Free-text matches ===")
    for q in ["ปวดหัว", "ขี้ลำบาก", "พูดเหมือนเมา", "หัวจะระเบิด", "ตด"]:
        m = match_freetext(q, dict_df)
        if m:
            print(f"  '{q}' → {m.symptom_en} ({m.label_th}) [{m.sublabel}]")
        else:
            print(f"  '{q}' → NO MATCH")

    print("\n=== Q3+ options (after picking ปวดท้อง + แสบยอดอก) ===")
    try:
        from utils.data_loader import get_scoring_artifacts
        arts = get_scoring_artifacts()
        opts = get_q3plus_options(["abdominal_pain", "acidity"], arts, dict_df, top_k=5)
        for o in opts:
            print(f"  • {o.label_th:30} ({o.symptom_en})  {o.sublabel}")

        print("\n=== is_tree_done ===")
        for n_picked, n_asked in [(1, 1), (2, 2), (3, 3), (5, 6)]:
            picks = ["abdominal_pain", "acidity", "nausea", "vomiting", "indigestion"][:n_picked]
            done = is_tree_done(picks, arts, n_asked)
            print(f"  picked={n_picked} asked={n_asked} → done={done}")
    except Exception as e:
        print(f"  (skipped Q3+ test: {e})")
