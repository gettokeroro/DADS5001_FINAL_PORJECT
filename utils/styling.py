"""
Global styling injector -- Phase 6 + Phase 8 polish
"""
from __future__ import annotations
from pathlib import Path
import base64
import streamlit as st

NAVY = "#02497E"
ACCENT = "#22B7E9"
PALE = "#EEFEFF"
PAGE_BG = "#F8FAFC"
BORDER = "#9DE0E9"

AI_BODY_TEXT    = "#042C53"
AI_SUB_TEXT     = "#185FA5"
AI_BORDER       = "#85B7EB"
AI_SCREEN_BG    = "#F4F8FC"
AI_DIAG_BG      = "#FDF4F8"
AI_HOVER_BG     = "#D4EDDA"
AI_HOVER_TXT    = "#1B5E20"
AI_DIAGNOSE_BTN = "#3DAA6D"
AI_NARRATION_BG = "rgba(255,255,255,0.85)"
AI_NARRATION_TXT = "#4B1528"

SIDEBAR_GRADIENT = (
    "linear-gradient(180deg,"
    "#EEFEFF 0%,"
    "#9DE0E9 30%,"
    "#22B7E9 60%,"
    "#3392D2 85%,"
    "#3891CE 100%)"
)
OVERLAY_ALPHA = 0.78


@st.cache_data(show_spinner=False)
def _bg_data_url() -> str:
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "BG_R1_web.jpg",
        project_root / "BG_R1.png",
    ]
    for p in candidates:
        if p.exists():
            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    return ""


def inject_global_css():
    bg = _bg_data_url()
    bg_layer = (
        f'linear-gradient(rgba(248,250,252,{OVERLAY_ALPHA}),'
        f'rgba(248,250,252,{OVERLAY_ALPHA})),'
        f'url("{bg}")'
    ) if bg else "none"

    css = (
        "<style>"
        "[data-testid=\"stMain\"],"
        "[data-testid=\"stAppViewContainer\"] > .main {"
        f"background-image:{bg_layer};"
        "background-size:cover;background-position:center;"
        "background-attachment:fixed;background-repeat:no-repeat;}"
        "[data-testid=\"stSidebar\"]{"
        f"background:{SIDEBAR_GRADIENT};}}"
        "[data-testid=\"stSidebar\"] *:not(button):not(a){"
        f"color:{NAVY} !important;}}"
        "[data-testid=\"stSidebar\"] [data-testid=\"stSidebarNav\"] a{"
        "background:rgba(255,255,255,0.45);border-radius:6px;"
        "margin-bottom:4px;padding:6px 10px;}"
        "[data-testid=\"stSidebar\"] [data-testid=\"stSidebarNav\"] a:hover{"
        "background:rgba(255,255,255,0.75);}"
        "[data-testid=\"stSidebar\"] [data-testid=\"stSidebarNav\"] a[aria-current=\"page\"]{"
        "background:rgba(255,255,255,0.92);font-weight:500;}"
        "[data-testid=\"stMain\"] p,[data-testid=\"stMain\"] span,"
        "[data-testid=\"stMain\"] li,[data-testid=\"stMain\"] label,"
        "[data-testid=\"stMain\"] h1,[data-testid=\"stMain\"] h2,"
        "[data-testid=\"stMain\"] h3,[data-testid=\"stMain\"] h4,"
        "[data-testid=\"stMain\"] h5,[data-testid=\"stMain\"] h6,"
        "[data-testid=\"stMain\"] [data-testid=\"stMarkdownContainer\"]{"
        f"color:{NAVY};}}"
        "[data-testid=\"stMain\"] [data-testid=\"stCaptionContainer\"],"
        "[data-testid=\"stMain\"] small,[data-testid=\"stMain\"] .stCaption{"
        f"color:{NAVY} !important;opacity:0.72;}}"
        "[data-testid=\"stMain\"] a{"
        f"color:{ACCENT};text-decoration:none;}}"
        "[data-testid=\"stMain\"] a:hover{"
        f"color:{NAVY};text-decoration:underline;}}"
        "[data-testid=\"stMain\"] [data-testid=\"stVerticalBlockBorderWrapper\"]{"
        "background:rgba(255,255,255,0.92);backdrop-filter:blur(2px);border-radius:8px;}"
        "[data-testid=\"stMain\"] [data-testid=\"stExpander\"] > details{"
        "background:rgba(255,255,255,0.88);backdrop-filter:blur(2px);"
        f"border-radius:8px;border:0.5px solid {BORDER};}}"
        "[data-testid=\"stHeader\"]{"
        "background:rgba(248,250,252,0.6);backdrop-filter:blur(4px);}"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


def inject_ai_mode_css() -> None:
    parts = [
        "<style>",
        f".ai-screen{{background:{AI_SCREEN_BG};border:1.5px solid {AI_BORDER};border-radius:12px;padding:22px 26px;margin-bottom:14px;}}",
        f".ai-screen.diagnosis{{background:{AI_DIAG_BG};}}",
        f".ai-screen p,.ai-screen span,.ai-screen label,.ai-screen li,.ai-screen h3,.ai-screen h4{{color:{AI_BODY_TEXT} !important;font-weight:500;}}",
        f".ai-screen .ai-sub{{color:{AI_SUB_TEXT} !important;font-size:13px;font-weight:400;}}",
        f".ai-mascot-box{{display:flex;align-items:center;gap:16px;background:rgba(255,255,255,0.72);border:1.5px solid {AI_BORDER};border-radius:12px;padding:12px 18px;margin-bottom:18px;}}",
        ".ai-mascot-box svg{flex-shrink:0;}",
        f".ai-mascot-speech{{color:{AI_BODY_TEXT} !important;font-size:15px;font-weight:600;line-height:1.55;}}",
        f".ai-mascot-speech .ai-sub{{display:block;font-weight:400;font-size:13px;color:{AI_SUB_TEXT} !important;margin-top:4px;}}",
        f".ai-tag{{display:inline-block;background:#E3F2FD;border:1px solid {AI_BORDER};border-radius:20px;padding:3px 11px;margin:3px 4px 3px 0;font-size:12px;color:{AI_BODY_TEXT} !important;font-weight:500;}}",
        f".ai-progress{{display:flex;align-items:center;gap:8px;margin-bottom:14px;}}",
        f".ai-dot{{width:9px;height:9px;border-radius:50%;background:{AI_BORDER};}}",
        ".ai-dot.done{background:#66BB6A;}",
        f".ai-dot.now{{background:{AI_SUB_TEXT};transform:scale(1.4);}}",
        f".ai-narration{{background:{AI_NARRATION_BG};border-left:4px solid #F48FB1;border-radius:0 8px 8px 0;padding:13px 17px;color:{AI_NARRATION_TXT} !important;font-size:14px;font-weight:500;line-height:1.6;margin-bottom:14px;}}",
        f"[data-testid=\"stMain\"] .stButton > button{{background:white !important;border:1.5px solid {AI_BORDER} !important;border-radius:8px !important;color:{AI_BODY_TEXT} !important;font-weight:500 !important;transition:background 0.15s,border-color 0.15s,color 0.15s;}}",
        f"[data-testid=\"stMain\"] .stButton > button:hover{{background:{AI_HOVER_BG} !important;border-color:#81C784 !important;color:{AI_HOVER_TXT} !important;}}",
        "[data-testid=\"stMain\"] .stButton > button:active{background:#C8E6C9 !important;}",
        f".diag-header{{background:#B5D4F4;color:{AI_BODY_TEXT};padding:12px 22px;border-radius:8px;margin-bottom:14px;text-align:center;font-size:15px;font-weight:600;border:1px solid {AI_BORDER};}}",
        f".diag-narrate{{background:white;padding:14px 16px;border-radius:8px;font-size:13.5px;line-height:1.75;border:1px solid #F4C0D1;margin-bottom:10px;color:{AI_BODY_TEXT};}}",
        f".diag-narrate h3{{font-size:14px;margin:0 0 7px;color:{AI_BODY_TEXT};font-weight:600;}}",
        ".diag-narrate .diag-quote{color:#4B1528;font-style:italic;line-height:1.7;}",
        ".mascot-sticky{position:sticky;top:80px;}",
        f".mascot-col-box{{background:#E6F1FB;border:1px solid {AI_BORDER};border-radius:12px;padding:14px;display:flex;flex-direction:column;align-items:center;text-align:center;gap:8px;}}",
        f".mascot-col-name{{font-size:13px;font-weight:600;color:{AI_BODY_TEXT};}}",
        f".mascot-col-line{{font-size:12px;color:{AI_SUB_TEXT};font-style:italic;line-height:1.5;}}",
        f".diagnose-btn button,[data-testid=\"stMain\"] .stButton.diagnose-btn > button{{background:{AI_DIAGNOSE_BTN} !important;border-color:{AI_DIAGNOSE_BTN} !important;color:white !important;font-weight:600 !important;font-size:15px !important;}}",
        f"[data-testid=\"stMain\"] .stButton.diagnose-btn > button:hover{{background:#2E8B57 !important;border-color:#2E8B57 !important;color:white !important;}}",
        f"[data-testid=\"stMain\"] .stTextInput input{{background:white !important;border:1.5px solid {AI_BORDER} !important;border-radius:8px !important;color:{AI_BODY_TEXT} !important;}}",
        "</style>",
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


_NURSE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="96" height="134">'
    '<defs>'
    '<radialGradient id="nc_face" cx="38%" cy="32%" r="66%">'
    '<stop offset="0%" stop-color="#FFF5F8"/>'
    '<stop offset="100%" stop-color="#FFBACF"/>'
    '</radialGradient>'
    '<radialGradient id="nc_body" cx="40%" cy="20%" r="80%">'
    '<stop offset="0%" stop-color="#FFDAEC"/>'
    '<stop offset="100%" stop-color="#FF8FB5"/>'
    '</radialGradient>'
    '<radialGradient id="nc_iris" cx="30%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#9B59B6"/>'
    '<stop offset="100%" stop-color="#2C1050"/>'
    '</radialGradient>'
    '</defs>'
    '<ellipse cx="60" cy="148" rx="26" ry="18" fill="url(#nc_body)" stroke="#FF80A8" stroke-width="1.2"/>'
    '<path d="M48 134 L60 144 L72 134" fill="white" stroke="#FFAAC4" stroke-width="0.8" opacity="0.85"/>'
    '<circle cx="60" cy="57" r="48" fill="url(#nc_face)" stroke="#FFC0D8" stroke-width="1.5"/>'
    '<ellipse cx="60" cy="13" rx="26" ry="7" fill="white" stroke="#E0D0DB" stroke-width="0.8"/>'
    '<rect x="46" y="7" width="28" height="10" rx="5" fill="white" stroke="#E0D0DB" stroke-width="0.8"/>'
    '<rect x="57.5" y="8.5" width="5" height="2" rx="1" fill="#E53935"/>'
    '<rect x="59.5" y="6.5" width="2" height="6" rx="1" fill="#E53935"/>'
    '<ellipse cx="41" cy="61" rx="13" ry="15" fill="white" stroke="#F0D0E0" stroke-width="0.6"/>'
    '<circle cx="41" cy="62.5" r="10" fill="url(#nc_iris)"/>'
    '<circle cx="41" cy="64" r="5.8" fill="#0C0418"/>'
    '<circle cx="46" cy="56.5" r="3.8" fill="white" opacity="0.95"/>'
    '<circle cx="36.5" cy="65.5" r="1.8" fill="white" opacity="0.7"/>'
    '<circle cx="44.5" cy="69.5" r="1.1" fill="white" opacity="0.55"/>'
    '<ellipse cx="79" cy="61" rx="13" ry="15" fill="white" stroke="#F0D0E0" stroke-width="0.6"/>'
    '<circle cx="79" cy="62.5" r="10" fill="url(#nc_iris)"/>'
    '<circle cx="79" cy="64" r="5.8" fill="#0C0418"/>'
    '<circle cx="84" cy="56.5" r="3.8" fill="white" opacity="0.95"/>'
    '<circle cx="74.5" cy="65.5" r="1.8" fill="white" opacity="0.7"/>'
    '<circle cx="82.5" cy="69.5" r="1.1" fill="white" opacity="0.55"/>'
    '<path d="M29 53 Q35 46 42 50" stroke="#4A1870" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
    '<path d="M78 50 Q85 46 91 53" stroke="#4A1870" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
    '<ellipse cx="24" cy="75" rx="11" ry="7" fill="#FF6090" opacity="0.25"/>'
    '<ellipse cx="96" cy="75" rx="11" ry="7" fill="#FF6090" opacity="0.25"/>'
    '<circle cx="60" cy="80" r="1.8" fill="#FFAAC0" opacity="0.65"/>'
    '<path d="M46 88 Q60 104 74 88" stroke="#E03060" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
    '<path d="M52 99 Q60 103 68 99" stroke="#E03060" stroke-width="1.3" fill="#FFD0E0" opacity="0.6"/>'
    '<path d="M60 158 C60 158 53 151 53 146 Q53 142 57 142 Q59.5 142 60 145 Q60.5 142 63 142 Q67 142 67 146 C67 151 60 158 60 158 Z" fill="#E53935"/>'
    '</svg>'
)

_DOCTOR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="96" height="134">'
    '<defs>'
    '<radialGradient id="dc_face" cx="38%" cy="32%" r="66%">'
    '<stop offset="0%" stop-color="#FFF9EE"/>'
    '<stop offset="100%" stop-color="#F8D898"/>'
    '</radialGradient>'
    '<radialGradient id="dc_body" cx="40%" cy="20%" r="80%">'
    '<stop offset="0%" stop-color="#F8F8F8"/>'
    '<stop offset="100%" stop-color="#DCDCDC"/>'
    '</radialGradient>'
    '<radialGradient id="dc_iris" cx="30%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#2E86AB"/>'
    '<stop offset="100%" stop-color="#0A2540"/>'
    '</radialGradient>'
    '</defs>'
    '<ellipse cx="60" cy="148" rx="28" ry="19" fill="url(#dc_body)" stroke="#C8C8C8" stroke-width="1.2"/>'
    '<path d="M48 134 L60 145 L72 134" fill="white" stroke="#E0E0E0" stroke-width="0.8"/>'
    '<path d="M54 134 L60 142" stroke="#CCC" stroke-width="0.7" fill="none"/>'
    '<path d="M66 134 L60 142" stroke="#CCC" stroke-width="0.7" fill="none"/>'
    '<circle cx="60" cy="57" r="48" fill="url(#dc_face)" stroke="#EED898" stroke-width="1.5"/>'
    '<path d="M16 50 Q18 22 60 18 Q102 22 104 50" fill="#5D4037" opacity="0.48"/>'
    '<circle cx="42" cy="64" r="13" fill="rgba(255,255,255,0.6)" stroke="#5D4037" stroke-width="2.2"/>'
    '<circle cx="78" cy="64" r="13" fill="rgba(255,255,255,0.6)" stroke="#5D4037" stroke-width="2.2"/>'
    '<line x1="55" y1="63" x2="65" y2="63" stroke="#5D4037" stroke-width="2.2"/>'
    '<line x1="29" y1="59" x2="21" y2="57" stroke="#5D4037" stroke-width="1.7"/>'
    '<line x1="91" y1="59" x2="99" y2="57" stroke="#5D4037" stroke-width="1.7"/>'
    '<circle cx="42" cy="65" r="9" fill="url(#dc_iris)"/>'
    '<circle cx="42" cy="66.5" r="5.2" fill="#050E20"/>'
    '<circle cx="46.5" cy="60" r="3.2" fill="white" opacity="0.92"/>'
    '<circle cx="38" cy="68.5" r="1.5" fill="white" opacity="0.65"/>'
    '<circle cx="78" cy="65" r="9" fill="url(#dc_iris)"/>'
    '<circle cx="78" cy="66.5" r="5.2" fill="#050E20"/>'
    '<circle cx="82.5" cy="60" r="3.2" fill="white" opacity="0.92"/>'
    '<circle cx="74" cy="68.5" r="1.5" fill="white" opacity="0.65"/>'
    '<ellipse cx="24" cy="78" rx="11" ry="7" fill="#FFAA88" opacity="0.26"/>'
    '<ellipse cx="96" cy="78" rx="11" ry="7" fill="#FFAA88" opacity="0.26"/>'
    '<circle cx="60" cy="81" r="1.8" fill="#C8906A" opacity="0.58"/>'
    '<path d="M46 90 Q60 106 74 90" stroke="#A06040" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
    '<path d="M52 101 Q60 105 68 101" stroke="#A06040" stroke-width="1.3" fill="#FFDDB0" opacity="0.6"/>'
    '<path d="M44 138 Q36 148 38 157 Q40 163 46 163 Q52 163 52 158" stroke="#78909C" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
    '<circle cx="52" cy="159" r="5.5" fill="#546E7A" stroke="#263238" stroke-width="1.2"/>'
    '<circle cx="52" cy="159" r="2.8" fill="#37474F"/>'
    '</svg>'
)


def render_nurse_mascot(speech: str, sub: str = "") -> None:
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="ai-mascot-box"><div style="width:88px;flex-shrink:0;line-height:0">' +
        _NURSE_SVG +
        f'</div><div class="ai-mascot-speech">{speech}{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_doctor_mascot(speech: str, sub: str = "") -> None:
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="ai-mascot-box"><div style="width:88px;flex-shrink:0;line-height:0">' +
        _DOCTOR_SVG +
        f'</div><div class="ai-mascot-speech">{speech}{sub_html}</div></div>',
        unsafe_allow_html=True,
    )
