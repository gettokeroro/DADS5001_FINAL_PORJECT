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

    # Disease prior — uniform (synthetic balanced)
    prior = {d: 1.0 / N for d in diseases}

    # DuckDB connection — register the DataFrames as tables
    conn = duckdb.connect(database=":memory:")
    conn.register("disease_symptom_long_df", disease_symptom_long)
    conn.register("symptom_idf_df", idf_df)
    # Persist as actual tables (zero-copy from pandas)
    conn.execute("CREATE TABLE disease_symptom_long AS SELECT * FROM disease_symptom_long_df")
    conn.execute("CREATE TABLE symptom_idf AS SELECT * FROM symptom_idf_df")
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
  ROUND(
    COALESCE(m.raw_score, 0)
    * (0.5 + 0.5 * COALESCE(m.n_matched * 1.0 / t.n_disease_symptoms, 0)),
    4
  ) AS score
FROM disease_total t
LEFT JOIN matched m ON t.disease = m.disease
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
  )
SELECT
  disease,
  ROUND(log_likelihood + LN(?), 3) AS log_posterior,
  n_matched
FROM per_disease
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

    prior = arts.disease_prior[arts.diseases[0]]  # uniform → same for all
    df = arts.duckdb_conn.execute(_BAYES_SQL, [valid, smooth, prior]).fetchdf()

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
        "name": "Tuberculosis — chronic cough with hemoptysis",
        "symptoms": ["cough", "chest_pain", "blood_in_sputum", "weight_loss",
                     "fatigue", "mild_fever", "sweating"],
        "expected_disease": "Tuberculosis",
    },
    {
        "name": "Dengue fever — classic presentation",
        "symptoms": ["high_fever", "headache", "joint_pain", "muscle_pain",
                     "red_spots_over_body", "vomiting", "fatigue", "pain_behind_the_eyes"],
        "expected_disease": "Dengue",
    },
    {
        "name": "Migraine — visual aura",
        "symptoms": ["headache", "nausea", "vomiting", "blurred_and_distorted_vision",
                     "indigestion"],
        "expected_disease": "Migraine",
    },
    {
        "name": "Diabetes — classic triad",
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
