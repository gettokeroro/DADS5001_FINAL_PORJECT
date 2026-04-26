"""
Symptom-to-Specialty scoring functions.

Provides 2 scoring strategies:
- TF-IDF style:  weights symptoms by specificity (rare symptoms = high weight)
- Bayesian:      P(disease | symptoms) using Naive Bayes assumption

Both functions return a ranked list of (disease, score, confidence) tuples.

Use from Streamlit:
    from utils.scoring import load_artifacts, predict
    arts = load_artifacts()                        # cached
    ranked = predict(user_symptoms, arts, method='tfidf')  # or 'bayes' or 'both'
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@dataclass
class ScoringArtifacts:
    """Pre-computed lookup tables used by both scoring methods."""
    diseases: list[str]                    # 41 unique disease names (with itachi trailing spaces preserved)
    symptoms: list[str]                    # 132 symptom column names
    disease_symptom_freq: pd.DataFrame     # rows = disease, cols = symptom, values = freq in [0,1]
    symptom_idf: dict[str, float]          # IDF weight per symptom
    disease_prior: dict[str, float]        # P(disease) — uniform here since synthetic balanced


def load_artifacts(data_dir: str | Path = "data") -> ScoringArtifacts:
    """Build all lookup tables from itachi_train.csv. Call once and cache."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "raw" / "itachi_train.csv")
    train = train.loc[:, ~train.columns.str.match(r"^Unnamed")]
    symptoms = [c for c in train.columns if c != "prognosis"]
    diseases = sorted(train["prognosis"].unique())

    # Disease × symptom frequency (P(symptom | disease))
    freq = train.groupby("prognosis")[symptoms].mean()  # mean of 0/1 = freq
    freq = freq.loc[diseases]

    # IDF: log(N / df) where df = #diseases that have this symptom (freq > 0)
    N = len(diseases)
    df_count = (freq > 0).sum(axis=0)  # how many diseases have this symptom
    idf = {s: math.log(N / max(df_count[s], 1)) for s in symptoms}

    # Disease prior — uniform since itachi is synthetically balanced (120 each)
    prior = {d: 1.0 / N for d in diseases}

    return ScoringArtifacts(
        diseases=diseases,
        symptoms=symptoms,
        disease_symptom_freq=freq,
        symptom_idf=idf,
        disease_prior=prior,
    )


# ---------------------------------------------------------------------------
# TF-IDF style scoring
# ---------------------------------------------------------------------------

def score_tfidf(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
) -> pd.DataFrame:
    """
    Score each disease against user-provided symptoms using TF-IDF style.

    Score(d) = Σ_{s in user ∩ disease_symptoms} freq(s|d) × idf(s)
               × coverage_bonus(disease symptoms matched)

    Returns DataFrame: disease, score, n_matched, n_disease_symptoms, coverage
    sorted by score desc.
    """
    if not user_symptoms:
        return pd.DataFrame(columns=["disease","score","n_matched","n_disease_symptoms","coverage"])

    valid = [s for s in user_symptoms if s in arts.symptoms]
    if not valid:
        return pd.DataFrame(columns=["disease","score","n_matched","n_disease_symptoms","coverage"])

    rows = []
    for d in arts.diseases:
        d_freq = arts.disease_symptom_freq.loc[d]
        d_symptoms_set = set(d_freq[d_freq > 0].index)
        matched = [s for s in valid if s in d_symptoms_set]

        # raw weighted sum
        raw_score = sum(d_freq[s] * arts.symptom_idf[s] for s in matched)

        # coverage = fraction of disease symptoms that user reported
        n_disease_syms = len(d_symptoms_set)
        coverage = len(matched) / n_disease_syms if n_disease_syms else 0.0

        # final score: weighted score × (0.5 + 0.5 × coverage) — favor diseases user reports comprehensively
        final = raw_score * (0.5 + 0.5 * coverage)

        rows.append({
            "disease": d,
            "score": round(final, 4),
            "n_matched": len(matched),
            "n_disease_symptoms": n_disease_syms,
            "coverage": round(coverage, 3),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    # Normalize to confidence (softmax over top-K)
    top_scores = df["score"].head(10).values
    if top_scores.sum() > 0:
        exp_scores = np.exp(top_scores - top_scores.max())
        confidence = exp_scores / exp_scores.sum()
        df["confidence"] = 0.0
        df.loc[df.index[:10], "confidence"] = np.round(confidence, 3)
    else:
        df["confidence"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Bayesian (Naive Bayes) scoring
# ---------------------------------------------------------------------------

def score_bayes(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
    smoothing: float = 0.01,
) -> pd.DataFrame:
    """
    P(disease | symptoms) ∝ P(disease) × Π P(symptom_i | disease)
    using log-space and Laplace-style smoothing for symptoms with freq=0.

    For symptoms NOT reported by user, we approximate by ignoring them
    (true Bayes would multiply by P(¬s|d)) — this keeps it simple and
    makes the score a "support level" rather than full posterior.

    Returns DataFrame: disease, log_posterior, posterior, n_matched
    sorted by posterior desc.
    """
    if not user_symptoms:
        return pd.DataFrame(columns=["disease","log_posterior","posterior","n_matched"])

    valid = [s for s in user_symptoms if s in arts.symptoms]
    if not valid:
        return pd.DataFrame(columns=["disease","log_posterior","posterior","n_matched"])

    rows = []
    for d in arts.diseases:
        d_freq = arts.disease_symptom_freq.loc[d]
        log_p = math.log(arts.disease_prior[d])
        n_match = 0
        for s in valid:
            p = d_freq[s]
            # smoothing: avoid log(0)
            p_smoothed = max(p, smoothing)
            log_p += math.log(p_smoothed)
            if p > 0:
                n_match += 1
        rows.append({
            "disease": d,
            "log_posterior": log_p,
            "n_matched": n_match,
        })

    df = pd.DataFrame(rows)
    # Normalize: subtract max log-posterior for stability, then softmax
    df["log_posterior"] = df["log_posterior"].astype(float)
    max_lp = df["log_posterior"].max()
    df["posterior"] = np.exp(df["log_posterior"] - max_lp)
    df["posterior"] = df["posterior"] / df["posterior"].sum()
    df["posterior"] = df["posterior"].round(4)
    df["log_posterior"] = df["log_posterior"].round(3)

    return df.sort_values("posterior", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified predict
# ---------------------------------------------------------------------------

def predict(
    user_symptoms: list[str],
    arts: ScoringArtifacts,
    method: str = "tfidf",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Predict top-K diseases from user symptoms.

    method: 'tfidf', 'bayes', or 'both' (returns merged ranking).
    Returns DataFrame with at least columns: disease, primary_score, rank.
    """
    if method == "tfidf":
        df = score_tfidf(user_symptoms, arts).head(top_k)
        df = df.rename(columns={"score": "primary_score"})
        df["method"] = "tfidf"
        df["rank"] = range(1, len(df) + 1)
        return df
    elif method == "bayes":
        df = score_bayes(user_symptoms, arts).head(top_k)
        df = df.rename(columns={"posterior": "primary_score"})
        df["method"] = "bayes"
        df["rank"] = range(1, len(df) + 1)
        return df
    elif method == "both":
        t = score_tfidf(user_symptoms, arts)
        b = score_bayes(user_symptoms, arts)
        merged = (
            t[["disease", "score", "coverage"]]
            .rename(columns={"score": "tfidf_score"})
            .merge(b[["disease", "posterior"]].rename(columns={"posterior": "bayes_posterior"}),
                   on="disease", how="outer")
        )
        # Combined rank: average rank from both methods
        merged["tfidf_rank"] = merged["tfidf_score"].rank(ascending=False, method="min")
        merged["bayes_rank"] = merged["bayes_posterior"].rank(ascending=False, method="min")
        merged["avg_rank"] = (merged["tfidf_rank"] + merged["bayes_rank"]) / 2
        merged = merged.sort_values("avg_rank").head(top_k).reset_index(drop=True)
        merged["rank"] = range(1, len(merged) + 1)
        merged["method"] = "both"
        return merged
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tfidf', 'bayes', or 'both'.")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(
    eval_cases: list[dict],
    arts: ScoringArtifacts,
    method: str = "tfidf",
) -> dict:
    """
    Evaluate scoring method against gold cases.

    eval_cases: list of {"name": str, "symptoms": [str], "expected_disease": str}

    Returns: { 'top1_hit_rate', 'top3_hit_rate', 'detail': [per-case results] }
    """
    detail = []
    top1 = top3 = 0
    for case in eval_cases:
        ranked = predict(case["symptoms"], arts, method=method, top_k=3)
        top_diseases = ranked["disease"].tolist()
        expected = case["expected_disease"]
        hit_top1 = top_diseases[0] == expected if top_diseases else False
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
    # Urgency 1 — Resuscitation
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
    # Urgency 2 — Emergent
    {
        "name": "Hypoglycemia in diabetic",
        "symptoms": ["sweating", "fatigue", "headache", "blurred_and_distorted_vision", "anxiety", "irritability"],
        "expected_disease": "Hypoglycemia",
    },
    {
        "name": "Malaria — fever with chills",
        "symptoms": ["high_fever", "chills", "headache", "vomiting", "sweating", "muscle_pain"],
        "expected_disease": "Malaria",
    },
    # Urgency 3 — Urgent
    {
        "name": "Pneumonia — productive cough + fever",
        "symptoms": ["cough", "high_fever", "breathlessness", "chest_pain", "phlegm", "fast_heart_rate", "rusty_sputum"],
        "expected_disease": "Pneumonia",
    },
    {
        "name": "Tuberculosis — chronic cough with hemoptysis",
        "symptoms": ["cough", "chest_pain", "blood_in_sputum", "weight_loss", "fatigue", "mild_fever", "sweating"],
        "expected_disease": "Tuberculosis",
    },
    {
        "name": "Dengue fever — classic presentation",
        "symptoms": ["high_fever", "headache", "joint_pain", "muscle_pain", "red_spots_over_body", "vomiting", "fatigue", "pain_behind_the_eyes"],
        "expected_disease": "Dengue",
    },
    # Urgency 4 — Less urgent
    {
        "name": "Migraine — visual aura",
        "symptoms": ["headache", "nausea", "vomiting", "blurred_and_distorted_vision", "indigestion"],
        "expected_disease": "Migraine",
    },
    {
        "name": "Diabetes — classic triad",
        "symptoms": ["fatigue", "polyuria", "increased_appetite", "weight_loss", "blurred_and_distorted_vision", "excessive_hunger", "irregular_sugar_level"],
        "expected_disease": "Diabetes ",  # itachi has trailing space
    },
    # Urgency 5 — Non-urgent
    {
        "name": "Common cold",
        "symptoms": ["runny_nose", "congestion", "throat_irritation", "cough", "continuous_sneezing", "chills"],
        "expected_disease": "Common Cold",
    },
]
