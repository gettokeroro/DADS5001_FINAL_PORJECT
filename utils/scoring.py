"""
Symptom-to-Specialty scoring engine — DuckDB SQL version.

ภายในใช้ DuckDB SQL queries (ไม่ใช่ pandas iteration) เพื่อตอบ requirement
บังคับของโจทย์ "Use DuckDB (pandas)"

Provides 2 scoring strategies:
- TF-IDF style:  weights symptoms by specificity (rare = high weight)
- Bayesian:      P(disease | symptoms) using Naive Bayes

Both run as DuckDB SQL queries.

Usage from Streamlit:
    from utils.scoring import load_artifacts, predict
    arts = load_artifacts()                        # cached
    ranked = predict(['high_fever','cough'], arts, method='tfidf')
    # ranked.attrs has engine info: 'engine', 'scoring_time_ms', 'rows_scanned'
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import time
import pandas as pd
import numpy as np
import duckdb


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@dataclass
class ScoringArtifacts:
    """Pre-computed lookup tables + DuckDB connection used by scoring methods."""
    diseases: list[str]                    # 41 unique disease names
    symptoms: list[str]                    # 132 symptom column names
    symptom_idf: dict[str, float]          # IDF weight per symptom
    disease_prior: dict[str, float]        # P(disease) — uniform on synthetic data
    duckdb_conn: duckdb.DuckDBPyConnection # DuckDB conn with tables registered
    n_diseases: int
    bayes_smoothing: float = 0.01


def load_artifacts(data_dir: str | Path = "data") -> ScoringArtifacts:
    """Build DuckDB tables + IDF lookup. Call once and cache via st.cache_resource."""
    data_dir = Path(data_dir)

    # Load training data with pandas first (DuckDB will get its data via this)
    train = pd.read_csv(data_dir / "raw" / "itachi_train.csv")
    train = train.loc[:, ~train.columns.str.match(r"^Unnamed")]
    symptoms = [c for c in train.columns if c != "prognosis"]
    diseases = sorted(train["prognosis"].unique())
    N = len(diseases)

    # Build long-form (disease, symptom, freq) — the table DuckDB will query
    freq = train.groupby("prognosis")[symptoms].mean()
    long_records = []
    for disease, row in freq.iterrows():
        for sym, f in row.items():
            if f > 0:
                long_records.append({"disease": disease, "symptom": sym, "freq": float(f)})
    disease_symptom_long = pd.DataFrame(long_records)

    # IDF: log(N / df) where df = #diseases that have this symptom
    df_count = (freq > 0).sum(axis=0)
    idf_records = [
        {"symptom": s, "idf": math.log(N / max(int(df_count[s]), 1)),
         "n_diseases": int(df_count[s])}
        for s in symptoms
    ]
    idf_df = pd.DataFrame(idf_records)
    idf_dict = {r["symptom"]: r["idf"] for r in idf_records}

    # === Population prevalence (Phase 1) ===
    # อิง Top OPD ranking 2564 ของไทย · ป้องกันการให้ Dengue/Malaria
    # น้ำหนักเท่ากับ Common Cold ในการตัดสินใจ
    prev_path = data_dir / "processed" / "disease_prevalence.csv"
    if prev_path.exists():
        prev_df = pd.read_csv(prev_path, encoding="utf-8-sig")
    else:
        # Fallback — uniform if file missing
        prev_df = pd.DataFrame({
            "disease_en": diseases,
            "prevalence_weight": [1.0] * N,
            "prevalence_tier": ["unknown"] * N,
        })

    # Disease prior — based on prevalence (replaces uniform)
    prev_dict = dict(zip(prev_df["disease_en"], prev_df["prevalence_weight"]))
    total_prev = sum(prev_dict.get(d, 1.0) for d in diseases)
    prior = {d: prev_dict.get(d, 1.0) / total_prev for d in diseases}

    # DuckDB connection — register the DataFrames as tables
    conn = duckdb.connect(database=":memory:")
    conn.register("disease_symptom_long_df", disease_symptom_long)
    conn.register("symptom_idf_df", idf_df)
    conn.register("prevalence_df", prev_df)
    # Persist as actual tables (zero-copy from pandas)
    conn.execute("CREATE TABLE disease_symptom_long AS SELECT * FROM disease_symptom_long_df")
    conn.execute("CREATE TABLE symptom_idf AS SELECT * FROM symptom_idf_df")
    conn.execute("CREATE TABLE prevalence AS SELECT * FROM prevalence_df")
    conn.execute("CREATE INDEX idx_dsl_symptom ON disease_symptom_long(symptom)")
    conn.execute("CREATE INDEX idx_dsl_disease ON disease_symptom_long(disease)")

    return ScoringArtifacts(
        diseases=diseases,
        symptoms=symptoms,
        symptom_idf=idf_dict,
        disease_prior=prior,
        duckdb_conn=conn,
        n_diseases=N,
    )


# ---------------------------------------------------------------------------
# TF-IDF scoring via DuckDB SQL
# ---------------------------------------------------------------------------

_TFIDF_SQL = """
WITH
  user_symptoms AS (
    SELECT UNNEST(?::VARCHAR[]) AS symptom
  ),
  disease_total AS (
    SELECT disease, COUNT(*) AS n_disease_symptoms
    FROM disease_symptom_long
    GROUP BY disease
  ),
  matched AS (
    SELECT
      l.disease,
      SUM(l.freq * i.idf) AS raw_score,
      COUNT(*)            AS n_matched
    FROM disease_symptom_long l
    JOIN user_symptoms u ON l.symptom = u.symptom
    JOIN symptom_idf  i ON l.symptom = i.symptom
    GROUP BY l.disease
  )
SELECT
  t.disease,
  COALESCE(m.raw_score, 0) AS raw_score,
  COALESCE(m.n_matched, 0) AS n_matched,
  t.n_disease_symptoms,
  ROUND(COALESCE(m.n_matched * 1.0 / t.n_disease_symptoms, 0), 3) AS coverage,
  COALESCE(p.prevalence_weight, 1.0) AS prevalence_weight,
  COALESCE(p.prevalence_tier, 'unknown') AS prevalence_tier,
  -- Prevalence-adjusted score (Phase 1 — softer formula):
  --   base_score = raw_score × (0.5 + 0.5 × coverage)
  --   prev_mult  = POWER(prev, 0.25)  -- subtle boost · CC=3.16x, HA=1.32x
  --   final      = base × prev_mult
  -- Floor prev at 0.5 so very rare diseases not penalized too much
  ROUND(
    COALESCE(m.raw_score, 0)
    * (0.5 + 0.5 * COALESCE(m.n_matched * 1.0 / t.n_disease_symptoms, 0))
    * POWER(GREATEST(COALESCE(p.prevalence_weight, 1.0), 0.5), 0.25),
    4
  ) AS score
FROM disease_total t
LEFT JOIN matched m ON t.disease = m.disease
LEFT JOIN prevalence p ON t.disease = p.disease_en
ORDER BY score DESC
"""


def score_tfidf(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
) -> pd.DataFrame:
    """Score every disease via TF-IDF in DuckDB SQL.
    Returns DataFrame: disease, score, n_matched, n_disease_symptoms, coverage, confidence.
    Result has .attrs['engine']='DuckDB', .attrs['scoring_time_ms'], .attrs['sql'].
    """
    t0 = time.perf_counter()
    valid = [s for s in user_symptoms if s in arts.symptoms]
    if not valid:
        empty = pd.DataFrame(columns=[
            "disease","score","n_matched","n_disease_symptoms","coverage","confidence"
        ])
        empty.attrs.update({"engine": "DuckDB", "scoring_time_ms": 0,
                            "rows_scanned": 0, "sql": _TFIDF_SQL})
        return empty

    df = arts.duckdb_conn.execute(_TFIDF_SQL, [valid]).fetchdf()

    # Confidence (softmax over top-10 scores)
    top_scores = df["score"].head(10).to_numpy(dtype=float)
    if top_scores.sum() > 0:
        ex = np.exp(top_scores - top_scores.max())
        confidence = ex / ex.sum()
        df["confidence"] = 0.0
        df.loc[df.index[:10], "confidence"] = np.round(confidence, 3)
    else:
        df["confidence"] = 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000
    df.attrs.update({
        "engine": "DuckDB",
        "scoring_time_ms": round(elapsed_ms, 2),
        "rows_scanned": int(len(df)),
        "sql": _TFIDF_SQL,
    })
    return df


# ---------------------------------------------------------------------------
# Bayes scoring via DuckDB SQL
# ---------------------------------------------------------------------------

_BAYES_SQL = """
WITH
  user_symptoms AS (SELECT UNNEST(?::VARCHAR[]) AS symptom),
  all_diseases AS (SELECT DISTINCT disease FROM disease_symptom_long),
  cross_pairs AS (
    SELECT d.disease, u.symptom
    FROM all_diseases d CROSS JOIN user_symptoms u
  ),
  with_freq AS (
    SELECT
      cp.disease,
      cp.symptom,
      COALESCE(l.freq, 0) AS freq
    FROM cross_pairs cp
    LEFT JOIN disease_symptom_long l
      ON cp.disease = l.disease AND cp.symptom = l.symptom
  ),
  per_disease AS (
    SELECT
      disease,
      SUM(LN(GREATEST(freq, ?))) AS log_likelihood,
      SUM(CASE WHEN freq > 0 THEN 1 ELSE 0 END) AS n_matched
    FROM with_freq
    GROUP BY disease
  ),
  prior_norm AS (
    -- Normalize prevalence to be valid prior P(disease)
    SELECT disease_en, prevalence_weight / SUM(prevalence_weight) OVER () AS prior_p
    FROM prevalence
  )
SELECT
  pd.disease,
  ROUND(pd.log_likelihood + LN(COALESCE(pn.prior_p, 1.0/41)), 3) AS log_posterior,
  pd.n_matched
FROM per_disease pd
LEFT JOIN prior_norm pn ON pd.disease = pn.disease_en
ORDER BY log_posterior DESC
"""


def score_bayes(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
    smoothing: float | None = None,
) -> pd.DataFrame:
    """P(disease | symptoms) via Naive Bayes in DuckDB SQL.
    Returns DataFrame: disease, log_posterior, posterior, n_matched.
    """
    t0 = time.perf_counter()
    smooth = smoothing if smoothing is not None else arts.bayes_smoothing
    valid = [s for s in user_symptoms if s in arts.symptoms]
    if not valid:
        empty = pd.DataFrame(columns=[
            "disease","log_posterior","posterior","n_matched"
        ])
        empty.attrs.update({"engine": "DuckDB", "scoring_time_ms": 0,
                            "rows_scanned": 0, "sql": _BAYES_SQL})
        return empty

    # Bayes now uses per-disease prior from prevalence table (Phase 1)
    df = arts.duckdb_conn.execute(_BAYES_SQL, [valid, smooth]).fetchdf()

    # Convert log_posterior → normalized posterior (softmax)
    max_lp = df["log_posterior"].max()
    df["posterior"] = np.exp(df["log_posterior"].astype(float) - max_lp)
    df["posterior"] = (df["posterior"] / df["posterior"].sum()).round(4)
    df = df.sort_values("posterior", ascending=False).reset_index(drop=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    df.attrs.update({
        "engine": "DuckDB",
        "scoring_time_ms": round(elapsed_ms, 2),
        "rows_scanned": int(len(df)),
        "sql": _BAYES_SQL,
    })
    return df


# ---------------------------------------------------------------------------
# Unified predict
# ---------------------------------------------------------------------------

def predict(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
    method: str = "tfidf",
    top_k: int = 3,
) -> pd.DataFrame:
    """Predict top-K diseases. method ∈ {'tfidf','bayes','both'}.
    Returns DataFrame with .attrs containing engine info."""
    if method == "tfidf":
        df = score_tfidf(user_symptoms, arts)
        attrs = df.attrs.copy()
        df = df.head(top_k).copy()
        df = df.rename(columns={"score": "primary_score"})
        df["method"] = "tfidf"
        df["rank"] = range(1, len(df) + 1)
        df.attrs.update(attrs)
        return df

    elif method == "bayes":
        df = score_bayes(user_symptoms, arts)
        attrs = df.attrs.copy()
        df = df.head(top_k).copy()
        df = df.rename(columns={"posterior": "primary_score"})
        df["method"] = "bayes"
        df["rank"] = range(1, len(df) + 1)
        df.attrs.update(attrs)
        return df

    elif method == "both":
        t = score_tfidf(user_symptoms, arts)
        b = score_bayes(user_symptoms, arts)
        merged = (
            t[["disease", "score", "coverage"]]
            .rename(columns={"score": "tfidf_score"})
            .merge(
                b[["disease", "posterior"]].rename(columns={"posterior": "bayes_posterior"}),
                on="disease", how="outer",
            )
        )
        merged["tfidf_rank"] = merged["tfidf_score"].rank(ascending=False, method="min")
        merged["bayes_rank"] = merged["bayes_posterior"].rank(ascending=False, method="min")
        merged["avg_rank"] = (merged["tfidf_rank"] + merged["bayes_rank"]) / 2
        merged = merged.sort_values("avg_rank").head(top_k).reset_index(drop=True)
        merged["rank"] = range(1, len(merged) + 1)
        merged["method"] = "both"
        # Combine timings
        merged.attrs.update({
            "engine": "DuckDB",
            "scoring_time_ms": round(
                t.attrs.get("scoring_time_ms", 0) + b.attrs.get("scoring_time_ms", 0), 2
            ),
            "rows_scanned": t.attrs.get("rows_scanned", 0) + b.attrs.get("rows_scanned", 0),
            "sql": "TF-IDF + Bayes (รวม 2 queries)",
        })
        return merged
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(
    eval_cases: list[dict],
    arts: ScoringArtifacts,
    method: str = "tfidf",
) -> dict:
    """Eval against gold cases. Returns dict with hit rates + per-case detail."""
    detail = []
    top1 = top3 = 0
    for case in eval_cases:
        ranked = predict(case["symptoms"], arts, method=method, top_k=3)
        top_diseases = ranked["disease"].tolist()
        expected = case["expected_disease"]
        hit_top1 = bool(top_diseases) and top_diseases[0] == expected
        hit_top3 = expected in top_diseases
        if hit_top1: top1 += 1
        if hit_top3: top3 += 1
        detail.append({
            "case": case["name"],
            "expected": expected,
            "top1": top_diseases[0] if top_diseases else None,
            "top3": top_diseases,
            "hit_top1": hit_top1,
            "hit_top3": hit_top3,
        })
    n = len(eval_cases) or 1
    return {
        "method": method,
        "top1_hit_rate": top1 / n,
        "top3_hit_rate": top3 / n,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Default eval set (10 cases · urgency 1-5)
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CASES = [
    {
        "name": "Heart attack — middle-aged male",
        "symptoms": ["chest_pain", "sweating", "breathlessness", "vomiting"],
        "expected_disease": "Heart attack",
    },
    {
        "name": "Stroke / brain hemorrhage",
        "symptoms": ["weakness_of_one_body_side", "slurred_speech", "headache", "altered_sensorium"],
        "expected_disease": "Paralysis (brain hemorrhage)",
    },
    {
        "name": "Hypoglycemia in diabetic",
        "symptoms": ["sweating", "fatigue", "headache", "blurred_and_distorted_vision",
                     "anxiety", "irritability"],
        "expected_disease": "Hypoglycemia",
    },
    {
        "name": "Malaria — fever with chills",
        "symptoms": ["high_fever", "chills", "headache", "vomiting", "sweating", "muscle_pain"],
        "expected_disease": "Malaria",
    },
    {
        "name": "Pneumonia — productive cough + fever",
        "symptoms": ["cough", "high_fever", "breathlessness", "chest_pain",
                     "phlegm", "fast_heart_rate", "rusty_sputum"],
        "expected_disease": "Pneumonia",
    },
    {
        "name": "Tuberculosis - chronic cough with hemoptysis",
        "symptoms": ["cough", "chest_pain", "blood_in_sputum", "weight_loss",
                     "fatigue", "mild_fever", "sweating"],
        "expected_disease": "Tuberculosis",
    },
    {
        "name": "Dengue fever - classic presentation",
        "symptoms": ["high_fever", "headache", "joint_pain", "muscle_pain",
                     "red_spots_over_body", "vomiting", "fatigue", "pain_behind_the_eyes"],
        "expected_disease": "Dengue",
    },
    {
        "name": "Migraine - visual aura",
        "symptoms": ["headache", "nausea", "vomiting", "blurred_and_distorted_vision",
                     "indigestion"],
        "expected_disease": "Migraine",
    },
    {
        "name": "Diabetes - classic triad",
        "symptoms": ["fatigue", "polyuria", "increased_appetite", "weight_loss",
                     "blurred_and_distorted_vision", "excessive_hunger",
                     "irregular_sugar_level"],
        "expected_disease": "Diabetes ",
    },
    {
        "name": "Common cold",
        "symptoms": ["runny_nose", "congestion", "throat_irritation", "cough",
                     "continuous_sneezing", "chills"],
        "expected_disease": "Common Cold",
    },
]


# ---------------------------------------------------------------------------
# Phase 1.5 — Co-symptom suggestion (interactive follow-up)
# ---------------------------------------------------------------------------

_CO_SYMPTOM_SQL = """
WITH
  user_symptoms AS (SELECT UNNEST(?::VARCHAR[]) AS symptom),
  -- Step A: rank candidate diseases lightly using just user input
  candidate_diseases AS (
    SELECT
      l.disease,
      SUM(l.freq * i.idf) * COALESCE(POWER(GREATEST(p.prevalence_weight, 0.5), 0.25), 1.0) AS rough_score
    FROM disease_symptom_long l
    JOIN user_symptoms u ON l.symptom = u.symptom
    JOIN symptom_idf i ON l.symptom = i.symptom
    LEFT JOIN prevalence p ON l.disease = p.disease_en
    GROUP BY l.disease, p.prevalence_weight
    ORDER BY rough_score DESC
    LIMIT 7
  ),
  -- Step B: find symptoms common in those candidates but NOT yet user-mentioned
  co_symptoms AS (
    SELECT
      l.symptom,
      SUM(l.freq * c.rough_score) AS relevance,
      COUNT(DISTINCT l.disease) AS n_diseases_have_it
    FROM disease_symptom_long l
    JOIN candidate_diseases c ON l.disease = c.disease
    WHERE l.symptom NOT IN (SELECT symptom FROM user_symptoms)
      AND l.freq >= 0.5
    GROUP BY l.symptom
    ORDER BY relevance DESC
    LIMIT ?
  )
SELECT
  cs.symptom,
  ROUND(cs.relevance, 3) AS relevance,
  cs.n_diseases_have_it
FROM co_symptoms cs
"""


def suggest_co_symptoms(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
    top_k: int = 5,
) -> pd.DataFrame:
    """Suggest top-K symptoms that commonly co-occur with user's initial symptoms.
    Used by AI Mode for interactive follow-up before final scoring.
    Returns DataFrame: symptom, relevance, n_diseases_have_it.
    """
    valid = [s for s in user_symptoms if s in arts.symptoms]
    if not valid:
        return pd.DataFrame(columns=["symptom", "relevance", "n_diseases_have_it"])
    return arts.duckdb_conn.execute(_CO_SYMPTOM_SQL, [valid, top_k]).fetchdf()


# ---------------------------------------------------------------------------
# Phase 2 — Confidence classification (combined: score + coverage + gap)
# ---------------------------------------------------------------------------

def classify_confidence(ranked_df: pd.DataFrame) -> dict:
    """Classify confidence of the ranked result based on multiple signals.

    Combines absolute score, gap to runner-up, and coverage to avoid the
    "softmax-confident-but-low-score" trap.

    Returns dict with:
        level: 'high' | 'medium' | 'low'
        emoji: visual indicator
        label: Thai display text
        reason: why this level
        top_score, gap_pct, top_coverage: raw signals (for debug)
    """
    if ranked_df.empty or len(ranked_df) == 0:
        return {
            "level": "low", "emoji": "🔴",
            "label": "ความมั่นใจต่ำมาก",
            "reason": "ไม่มีโรคใดในฐานข้อมูลตรงกับอาการที่ระบุ",
            "top_score": 0.0, "gap_pct": 0.0, "top_coverage": 0.0,
        }

    top_score = float(ranked_df["score"].iloc[0]) if "score" in ranked_df.columns \
                else float(ranked_df["primary_score"].iloc[0])
    second_score = (
        float(ranked_df["score"].iloc[1]) if len(ranked_df) > 1 and "score" in ranked_df.columns
        else float(ranked_df["primary_score"].iloc[1]) if len(ranked_df) > 1
        else 0.0
    )
    top_coverage = float(ranked_df["coverage"].iloc[0]) if "coverage" in ranked_df.columns else 0.0
    gap_pct = ((top_score - second_score) / top_score) if top_score > 0 else 0.0

    # Combined thresholds — high if (score strong AND coverage good) OR (score very strong AND gap clear)
    is_high = (top_score > 5 and top_coverage > 0.30) or (top_score > 8 and gap_pct > 0.50)
    if is_high:
        return {
            "level": "high", "emoji": "🟢",
            "label": "ความมั่นใจสูง",
            "reason": f"คะแนน {top_score:.1f} สูง · ครอบคลุมอาการ {top_coverage:.0%} · "
                      f"ทิ้งห่างอันดับ 2 ที่ {gap_pct:.0%}",
            "top_score": top_score, "gap_pct": gap_pct, "top_coverage": top_coverage,
        }
    if (top_score > 2 and top_coverage > 0.15) or (top_score > 4):
        return {
            "level": "medium", "emoji": "🟡",
            "label": "ความมั่นใจปานกลาง",
            "reason": f"คะแนน {top_score:.1f} พอใช้ · ครอบคลุม {top_coverage:.0%} · "
                      "ระบบเชื่อแต่ไม่แน่ใจมาก ควรพิจารณาผลรวมๆ",
            "top_score": top_score, "gap_pct": gap_pct, "top_coverage": top_coverage,
        }
    return {
        "level": "low", "emoji": "🔴",
        "label": "ความมั่นใจต่ำ",
        "reason": f"คะแนน {top_score:.1f} ต่ำ หรือครอบคลุมแค่ {top_coverage:.0%} - อาการกว้างเกินไป ระบบเดาอย่างมั่นใจไม่ได้ - แนะนำให้บอกอาการเพิ่มเติม",
        "top_score": top_score, "gap_pct": gap_pct, "top_coverage": top_coverage,
    }
