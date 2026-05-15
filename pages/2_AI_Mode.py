"""
Page 2 · AI Mode — Phase 8 redesign
=====================================
UX: button-based question tree (Q1→Q2→Q3+→diagnosis)
  Q1  chief complaint (6 buckets)
  Q2  sub-symptoms of that bucket
  Q3+ adaptive co-symptoms from suggest_co_symptoms
  free text screen — match กับ symptom_dictionary_th
  diagnosis — doctor mascot · Gemma narrate · drug + hospital cards

Architecture: Hybrid (symptom tree + Gemma narrate ตอนจบ)
State storage: st.session_state["ai8_*"]
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import (
    get_scoring_artifacts,
    init_session_state,
    load_drug_mapping,
    load_hospital_hint,
    load_hospitals_master,
    load_specialty_keywords,
    load_specialty_mapping,
    load_symptom_dict,
    render_disclaimer_sidebar,
    render_drug_panel,
    render_hospital_panel,
)
from utils.ai_engine import (
    AIResult,
    ExtractedSymptoms,
    _resolve_model_name,
    check_rate_limit,
    list_available_models,
    narrate_result,
    narrate_cards,
)
from utils.scoring import (
    classify_confidence,
    predict,
    score_tfidf,
    suggest_co_symptoms,
)
from utils.styling import (
    _DOCTOR_SVG,
    inject_ai_mode_css,
    inject_global_css,
    render_doctor_mascot,
    render_nurse_mascot,
)
from utils.symptom_tree import (
    CHIEF_COMPLAINTS,
    FREETEXT_CODE,
    SKIP_CODE,
    TreeOption,
    get_q1_options,
    get_q2_options,
    get_q3plus_options,
    is_tree_done,
    match_freetext,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Mode — น้องอุ่นใน", page_icon="🏥", layout="wide")
inject_global_css()
inject_ai_mode_css()
init_session_state()
render_disclaimer_sidebar()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CALLS      = 20
TOP_K_Q3       = 5
ICONS = {
    "fever":           "🤒",
    "cold":            "🤧",
    "headache":        "🤕",
    "stomach":         "🤢",
    "musculoskeletal": "💪",
    "skin":            "🩹",
}

# ---------------------------------------------------------------------------
# Session state keys (ai8_* prefix to avoid clash with old ai_ keys)
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "ai8_step":            "q1",   # q1|q2|q3plus|freetext|result
    "ai8_chief":           None,   # dict from CHIEF_COMPLAINTS
    "ai8_picked":          [],     # list[str] — accumulated symptom_en codes
    "ai8_seen":            set(),  # symptom_en already offered (for Q3+ exclusion)
    "ai8_q_count":         0,      # number of question screens shown (for soft cap)
    "ai8_freetext_prev":   "q2",   # step to return after freetext resolves
    "ai8_call_counter":    0,
    "ai8_narration":       "",
    "ai8_ranked":          None,
    "ai8_confidence":      None,
    "ai8_prov_key":        "ai8_provinces",  # key for province multiselect widget
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


def _api_key():
    try:
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _progress_dots(current_step: str) -> str:
    steps = ["q1", "q2", "q3plus", "result"]
    dots = []
    for s in steps:
        if s == current_step:
            dots.append('<span class="ai-dot now"></span>')
        elif steps.index(s) < steps.index(current_step):
            dots.append('<span class="ai-dot done"></span>')
        else:
            dots.append('<span class="ai-dot"></span>')
    return f'<div class="ai-progress">{"".join(dots)}</div>'


def _tag_html(picked: list[str], sym_dict: pd.DataFrame) -> str:
    tags = []
    for code in picked:
        match = sym_dict[sym_dict["symptom_en"] == code]
        th = match["symptom_th"].iloc[0] if not match.empty else code
        tags.append(f'<span class="ai-tag">{th}</span>')
    return " ".join(tags)


# ---------------------------------------------------------------------------
# Load shared data (cached)
# ---------------------------------------------------------------------------
sym_dict  = load_symptom_dict()
mapping   = load_specialty_mapping()
arts      = get_scoring_artifacts()
drug_df   = load_drug_mapping()
hint_df   = load_hospital_hint()
hosp_df   = load_hospitals_master()
kw_dict   = load_specialty_keywords()
vis_dict  = sym_dict[sym_dict["is_user_facing"] == True].copy()

api_key = _api_key()

# ---------------------------------------------------------------------------
# Sidebar — rate limit counter + reset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    calls_used = st.session_state.ai8_call_counter
    st.metric("AI Calls", f"{calls_used}/{MAX_CALLS}")
    if st.button("🔄 เริ่มใหม่", use_container_width=True):
        _reset()

if not api_key:
    st.error(
        "⚠ ยังไม่ได้ตั้งค่า GOOGLE_API_KEY · "
        "ขอ key ที่ aistudio.google.com แล้วใส่ใน Streamlit Cloud → Settings → Secrets"
    )
    st.stop()

# ===========================================================================
# Q1 — Chief complaint
# ===========================================================================
if st.session_state.ai8_step == "q1":
    st.markdown(_progress_dots("q1"), unsafe_allow_html=True)

    render_nurse_mascot(
        "สวัสดีค่ะ หนูชื่อ น้องอุ่นใน 😊",
        sub="วันนี้พี่มีอาการอะไรมาคะ? เลือกที่ตรงกับที่พี่เป็นได้เลยนะคะ",
    )

    options = get_q1_options()
    col_pairs = [options[i:i+2] for i in range(0, len(options), 2)]
    for pair in col_pairs:
        cols = st.columns(len(pair))
        for col, cc in zip(cols, pair):
            icon = ICONS.get(cc["id"], "🏥")
            label = f"{icon} {cc['label']}"
            if col.button(label, use_container_width=True, key=f"q1_{cc['id']}"):
                st.session_state.ai8_chief   = cc
                st.session_state.ai8_q_count = 1
                st.session_state.ai8_step    = "q2"
                st.rerun()

    st.markdown("---")
    st.caption("หรือถ้าพี่มีอาการอื่นที่ไม่ตรงกับด้านบน พิมพ์ในช่องด้านล่างได้เลยค่ะ")
    with st.expander("✏️ พิมพ์อาการเอง"):
        ft_in = st.text_input("พิมพ์อาการ (ภาษาไทยหรืออังกฤษ)", key="q1_freetext_in")
        if st.button("ตรวจสอบ", key="q1_freetext_check"):
            ft_in = ft_in.strip()
            if ft_in:
                match = match_freetext(ft_in, vis_dict)
                if match:
                    st.success(f"✅ พบ: {match.label_th} ({match.symptom_en})")
                    st.session_state.ai8_picked.append(match.symptom_en)
                    st.session_state.ai8_seen.add(match.symptom_en)
                    st.session_state.ai8_freetext_prev = "q1"
                    st.session_state.ai8_step = "q3plus"
                    st.session_state.ai8_q_count += 1
                    st.rerun()
                else:
                    st.warning("ไม่พบอาการที่ตรงกัน — ลองพิมพ์ด้วยคำอื่นนะคะ (เช่น ปวดหัว, ไอ, คลื่นไส้)")


# ===========================================================================
# Q2 — Sub-symptoms of chief complaint
# ===========================================================================
elif st.session_state.ai8_step == "q2":
    cc   = st.session_state.ai8_chief
    st.markdown(_progress_dots("q2"), unsafe_allow_html=True)

    prompt = cc.get("q2_prompt", "อาการไหนเด่นคะ?")
    render_nurse_mascot(
        f"พี่เลือก **{cc['label']}** ค่ะ 😊",
        sub=f"{prompt} (เลือกได้หลายข้อ)",
    )

    # Show basket if already picked something
    if st.session_state.ai8_picked:
        st.markdown(
            "**อาการที่เลือกแล้ว:** " + _tag_html(st.session_state.ai8_picked, sym_dict),
            unsafe_allow_html=True,
        )

    opts = get_q2_options(cc["id"], vis_dict)
    st.markdown("##### เลือกอาการที่พี่รู้สึก:")

    for opt in opts:
        if opt.symptom_en == FREETEXT_CODE:
            continue  # shown separately below
        icon = "✅" if opt.symptom_en in st.session_state.ai8_picked else "⬜"
        if st.button(f"{icon} {opt.label_th}", key=f"q2_{opt.symptom_en}", use_container_width=True):
            picked = st.session_state.ai8_picked
            if opt.symptom_en not in picked:
                picked.append(opt.symptom_en)
            st.session_state.ai8_seen.add(opt.symptom_en)
            # Always go to Q3+ — user decides when to stop
            st.session_state.ai8_step = "q3plus"
            st.session_state.ai8_q_count += 1
            st.rerun()

    # Freetext option
    with st.expander("✏️ อื่นๆ พิมพ์เอง"):
        ft_in = st.text_input("พิมพ์อาการ", key="q2_freetext_in")
        if st.button("ตรวจสอบ", key="q2_freetext_check"):
            ft_in = ft_in.strip()
            if ft_in:
                match = match_freetext(ft_in, vis_dict)
                if match:
                    st.success(f"✅ พบ: {match.label_th}")
                    if match.symptom_en not in st.session_state.ai8_picked:
                        st.session_state.ai8_picked.append(match.symptom_en)
                    st.session_state.ai8_seen.add(match.symptom_en)
                    st.session_state.ai8_freetext_prev = "q2"
                    st.session_state.ai8_q_count += 1
                    st.session_state.ai8_step = "q3plus"
                    st.rerun()
                else:
                    st.warning("ไม่พบอาการที่ตรงกัน — ลองพิมพ์ด้วยคำอื่นนะคะ")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← กลับ", key="q2_back", use_container_width=True):
            st.session_state.ai8_step = "q1"
            st.rerun()
    with col_b:
        picked_count = len(st.session_state.ai8_picked)
        btn_label = f"ขอคำวินิจฉัยเลย ({picked_count} อาการ)" if picked_count else "ข้ามไปวินิจฉัย"
        if st.button(btn_label, key="q2_skip", use_container_width=True):
            if picked_count == 0:
                # Add chief complaint seed symptoms
                for s in cc["symptoms"][:2]:
                    if s not in st.session_state.ai8_picked:
                        st.session_state.ai8_picked.append(s)
            st.session_state.ai8_step = "result"
            st.rerun()


# ===========================================================================
# Q3+ — Adaptive co-symptom questions
# ===========================================================================
elif st.session_state.ai8_step == "q3plus":
    picked   = st.session_state.ai8_picked
    q_count  = st.session_state.ai8_q_count
    st.markdown(_progress_dots("q3plus"), unsafe_allow_html=True)

    render_nurse_mascot(
        "มีอาการอื่นร่วมด้วยไหมคะ? 🩺",
        sub="หนูค้นหาอาการที่มักเกิดร่วมกันให้แล้วค่ะ",
    )

    # Basket
    if picked:
        st.markdown(
            "**อาการที่เลือกแล้ว:** " + _tag_html(picked, sym_dict),
            unsafe_allow_html=True,
        )

    opts = get_q3plus_options(picked, arts, vis_dict, top_k=TOP_K_Q3, exclude=st.session_state.ai8_seen)

    st.markdown("##### มีอาการข้างล่างนี้ด้วยไหมคะ?")
    for opt in opts:
        if opt.symptom_en == FREETEXT_CODE:
            continue

        if opt.is_skip:   # "ไม่มีอาการอื่น"
            if st.button(f"✅ {opt.label_th}", key=f"q3_skip_{q_count}", use_container_width=True):
                st.session_state.ai8_step = "result"
                st.rerun()
        else:
            icon = "✅" if opt.symptom_en in picked else "⬜"
            if st.button(
                f"{icon} {opt.label_th}",
                key=f"q3_{opt.symptom_en}_{q_count}",
                use_container_width=True,
                help=opt.sublabel or "",
            ):
                if opt.symptom_en not in picked:
                    picked.append(opt.symptom_en)
                st.session_state.ai8_seen.add(opt.symptom_en)
                st.session_state.ai8_q_count += 1
                # Stay in Q3+ — user controls exit via "ไม่มีอาการอื่น" or diagnose button
                st.session_state.ai8_step = "q3plus"
                st.rerun()

    # Freetext option
    with st.expander("✏️ อื่นๆ พิมพ์เอง"):
        ft_in = st.text_input("พิมพ์อาการ", key=f"q3_freetext_in_{q_count}")
        if st.button("ตรวจสอบ", key=f"q3_freetext_check_{q_count}"):
            ft_in = ft_in.strip()
            if ft_in:
                match = match_freetext(ft_in, vis_dict)
                if match:
                    st.success(f"✅ พบ: {match.label_th}")
                    if match.symptom_en not in picked:
                        picked.append(match.symptom_en)
                    st.session_state.ai8_seen.add(match.symptom_en)
                    st.session_state.ai8_q_count += 1
                    st.session_state.ai8_step = "q3plus"
                    st.rerun()
                else:
                    st.warning("ไม่พบอาการที่ตรงกัน — ลองพิมพ์ด้วยคำอื่นนะคะ")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← กลับ", key=f"q3_back_{q_count}", use_container_width=True):
            st.session_state.ai8_step = "q2"
            st.rerun()
    with col_b:
        if st.button(
            f"🏥 ขอคำวินิจฉัย ({len(picked)} อาการ)",
            key=f"q3_diagnose_{q_count}",
            use_container_width=True,
        ):
            st.session_state.ai8_step = "result"
            st.rerun()


# ===========================================================================
# RESULT — Scoring + card narrations + Top-3 + Drug/Hospital (mockup v4)
# ===========================================================================
elif st.session_state.ai8_step == "result":
    picked = st.session_state.ai8_picked

    # Guard: ถ้าไม่มีอาการเลย ส่งกลับ Q1
    if not picked:
        st.warning("ยังไม่มีอาการที่เลือก — กรุณาเริ่มต้นใหม่")
        if st.button("← กลับหน้าแรก"):
            _reset()
        st.stop()

    st.markdown(_progress_dots("result"), unsafe_allow_html=True)

    # Province filter — in main content so user can find it easily
    all_provinces = (
        sorted(hosp_df["province"].dropna().unique().tolist())
        if not hosp_df.empty else []
    )
    selected_prov = st.multiselect(
        "🏥 กรองจังหวัด (ค้นหาโรงพยาบาลใกล้บ้าน)",
        options=all_provinces,
        default=[],
        key=st.session_state.ai8_prov_key,
        placeholder="ทุกจังหวัด — เลือกเพื่อดู รพ. ในจังหวัดของคุณ",
    )

    # ── Scoring + card narrations (run once, cache in session_state) ─────────
    if st.session_state.ai8_ranked is None:
        allowed, _ = check_rate_limit(
            st.session_state, max_calls=MAX_CALLS, counter_key="ai8_call_counter"
        )
        if not allowed:
            st.error(f"Rate limit ถึงแล้ว ({MAX_CALLS} calls/session) · กด 🔄 เริ่มใหม่ด้านซ้าย")
            st.stop()

        with st.spinner("🏥 กำลังวิเคราะห์อาการ..."):
            ranked  = predict(picked, arts, method="tfidf", top_k=3)
            full_sc = score_tfidf(picked, arts)
            conf    = classify_confidence(full_sc, n_user_symptoms=len(picked))

            # Build Thai symptom labels for narration prompt
            picked_th = []
            for code in picked:
                m = sym_dict[sym_dict["symptom_en"] == code]
                picked_th.append(m["symptom_th"].iloc[0] if not m.empty else code)

            # Card narrations via Gemma (JSON → list[str])
            allowed2, _ = check_rate_limit(
                st.session_state, max_calls=MAX_CALLS, counter_key="ai8_call_counter"
            )
            if allowed2:
                try:
                    narrations, _ = narrate_cards(picked_th, ranked, mapping, api_key)
                except Exception:
                    narrations = ["", "", ""]
            else:
                narrations = ["", "", ""]

        st.session_state.ai8_ranked     = ranked
        st.session_state.ai8_confidence = conf
        st.session_state.ai8_narration  = narrations  # now list[str]

    # Pull from cache
    ranked     = st.session_state.ai8_ranked
    conf       = st.session_state.ai8_confidence
    narrations = st.session_state.ai8_narration
    if isinstance(narrations, str):          # backward compat with old string format
        narrations = [narrations, "", ""]

    # ── Blue header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="diag-header">🩺 คำวินิจฉัยโรคของคุณคือ :</div>',
        unsafe_allow_html=True,
    )

    # Symptom basket
    st.markdown(
        "**อาการที่วิเคราะห์:** " + _tag_html(picked, sym_dict),
        unsafe_allow_html=True,
    )

    # Confidence banners
    if conf["level"] == "very_low":
        st.warning(
            "⚠️ **ระบบไม่สามารถสรุปได้ชัดเจน** — อาการน้อยเกินไป "
            "ผลด้านล่างใช้เป็น reference เท่านั้น · กรุณาปรึกษาแพทย์"
        )
    elif conf["level"] == "low":
        st.info(
            "ℹ️ **ผลใช้เป็น reference ประกอบการตัดสินใจ** — ปรึกษาแพทย์เพื่อความแน่ใจ"
        )

    # ── Two-column layout: disease cards (left) + doctor mascot (right) ──────
    col_main, col_mascot = st.columns([3, 1], gap="medium")

    # ── RIGHT: Doctor mascot ─────────────────────────────────────────────────
    with col_mascot:
        _conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "very_low": "🔴"}
        _emoji = _conf_emoji.get(conf["level"], "🔵")
        st.markdown(
            f"""<div class="mascot-col-box">
                  <div style="line-height:0">{_DOCTOR_SVG}</div>
                  <div class="mascot-col-name">คุณหมอใจดี</div>
                  <div class="mascot-col-line">"ไม่ต้องกังวลนะคะ มาฟังกัน"</div>
                  <div class="mascot-col-line" style="margin-top:4px">
                    ความมั่นใจ {_emoji} {conf['label']}<br>
                    <span style="font-size:11px">{conf['reason']}</span>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── LEFT: Disease narration cards ────────────────────────────────────────
    with col_main:
        if ranked is not None and not ranked.empty:
            enriched = ranked.merge(
                mapping, left_on="disease", right_on="disease_en", how="left"
            )

            URGENCY_LABEL = {
                1: "🟥 ระดับ 1 — ฉุกเฉินทันที",
                2: "🟧 ระดับ 2 — รีบเข้า รพ.",
                3: "🟨 ระดับ 3 — ภายใน 24 ชม.",
                4: "🟦 ระดับ 4 — นัดตามคิว",
                5: "🟩 ระดับ 5 — ไม่เร่งด่วน",
            }

            # Normalize top-3 scores to proportional % (TF-IDF scores are unbounded)
            _top3_df = enriched.head(3)
            _raw = [float(_top3_df.iloc[j].get("primary_score", 0) or 0) for j in range(len(_top3_df))]
            _total = sum(_raw) if sum(_raw) > 0 else 1.0
            _norm_pcts = [s / _total for s in _raw]

            for i, (_, row) in enumerate(enriched.head(3).iterrows()):
                rank_num   = i + 1
                disease_th = row.get("disease_th") or row["disease"]
                disease_en = row.get("disease_en") or row["disease"]
                pct        = f"{_norm_pcts[i] * 100:.0f}%"
                narr       = narrations[i] if i < len(narrations) else ""
                primary    = row.get("primary_specialty") or "—"
                urg        = int(row["urgency_level"]) if pd.notna(row.get("urgency_level")) else 5
                red_flags  = row.get("red_flags") or ""
                disease_key = disease_en

                # Narration card HTML
                narr_html = (
                    f'<div class="diag-quote">"{narr}"</div>' if narr
                    else ""
                )
                st.markdown(
                    f"""<div class="diag-narrate">
                          <h3>{rank_num}. {disease_th} — โอกาส {pct}</h3>
                          {narr_html}
                        </div>""",
                    unsafe_allow_html=True,
                )

                # Red flags warning
                if red_flags:
                    st.warning(f"⚠ Red flags: {red_flags}", icon="⚠️")

                # Drug + Hospital — stacked vertically (minimal · easier to read when expanded)
                render_drug_panel(disease_key, drug_df)
                if primary and primary != "—":
                    render_hospital_panel(
                        primary,
                        hint_df,
                        hospitals_df=hosp_df,
                        keywords_dict=kw_dict,
                        selected_provinces=selected_prov if selected_prov else None,
                        key_suffix=f"_ai8_r{rank_num}_{disease_key}",
                    )

                # Urgency badge below each card
                st.caption(
                    f"🏥 แผนก: **{primary}** · {URGENCY_LABEL.get(urg, f'ระดับ {urg}')}"
                )

                if rank_num < 3:
                    st.markdown("")  # visual spacer between cards

    # ── Bottom: action buttons ───────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("← ถามใหม่", use_container_width=True):
            _reset()
    with col_b:
        with st.expander("🔧 Debug"):
            if st.session_state.ai8_ranked is not None:
                cols_show = [c for c in
                             ["disease", "primary_score", "n_matched", "coverage"]
                             if c in ranked.columns]
                st.dataframe(ranked[cols_show], use_container_width=True, hide_index=True)
            st.json({"picked": picked, "q_count": st.session_state.ai8_q_count,
                     "calls": st.session_state.ai8_call_counter})
