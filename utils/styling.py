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
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="88" height="124">'
    '<defs>'
    '<radialGradient id="ng_face" cx="42%" cy="36%" r="64%">'
    '<stop offset="0%" stop-color="#FFF2F6"/>'
    '<stop offset="100%" stop-color="#F9C4D6"/>'
    '</radialGradient>'
    '<radialGradient id="ng_body" cx="40%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#FDE8F0"/>'
    '<stop offset="100%" stop-color="#F0A8C0"/>'
    '</radialGradient>'
    '</defs>'
    '<ellipse cx="60" cy="114" rx="40" ry="52" fill="url(#ng_body)" stroke="#E0A0B8" stroke-width="1.2"/>'
    '<path d="M42 102 L60 120 L78 102" fill="white" stroke="#DDD" stroke-width="1"/>'
    '<circle cx="60" cy="62" r="40" fill="url(#ng_face)" stroke="#EEB4C8" stroke-width="1.5"/>'
    '<rect x="28" y="26" width="64" height="12" rx="6" fill="white" stroke="#D0D0D0" stroke-width="1"/>'
    '<rect x="34" y="15" width="52" height="15" rx="6" fill="white" stroke="#D0D0D0" stroke-width="1"/>'
    '<rect x="57" y="17.5" width="6" height="2.5" rx="1.2" fill="#E53935"/>'
    '<rect x="59.5" y="15" width="2.5" height="8" rx="1.2" fill="#E53935"/>'
    '<ellipse cx="45" cy="68" rx="9" ry="10" fill="white" stroke="#E0C0D0" stroke-width="0.8"/>'
    '<circle cx="45" cy="69" r="6" fill="#2C1B35"/>'
    '<circle cx="47.5" cy="66" r="2.3" fill="white" opacity="0.9"/>'
    '<circle cx="42.5" cy="71" r="1.1" fill="white" opacity="0.5"/>'
    '<ellipse cx="75" cy="68" rx="9" ry="10" fill="white" stroke="#E0C0D0" stroke-width="0.8"/>'
    '<circle cx="75" cy="69" r="6" fill="#2C1B35"/>'
    '<circle cx="77.5" cy="66" r="2.3" fill="white" opacity="0.9"/>'
    '<circle cx="72.5" cy="71" r="1.1" fill="white" opacity="0.5"/>'
    '<path d="M37 61 Q41 57 46 59.5" stroke="#5B2D60" stroke-width="1.4" fill="none" stroke-linecap="round"/>'
    '<path d="M74 59.5 Q79 57 83 61" stroke="#5B2D60" stroke-width="1.4" fill="none" stroke-linecap="round"/>'
    '<ellipse cx="30" cy="80" rx="11" ry="7" fill="#F48FB1" opacity="0.32"/>'
    '<ellipse cx="90" cy="80" rx="11" ry="7" fill="#F48FB1" opacity="0.32"/>'
    '<ellipse cx="60" cy="80" rx="2.5" ry="1.8" fill="#E898B0" opacity="0.55"/>'
    '<path d="M49 91 Q60 103 71 91" stroke="#C04070" stroke-width="2.3" fill="none" stroke-linecap="round"/>'
    '<path d="M60 135 C60 135 51 127 51 120 Q51 115 56.5 115 Q59 115 60 118 Q61 115 63.5 115 Q69 115 69 120 C69 127 60 135 60 135 Z" fill="#E53935" opacity="0.88"/>'
    '</svg>'
)

_DOCTOR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="88" height="124">'
    '<defs>'
    '<radialGradient id="dg_face" cx="42%" cy="36%" r="64%">'
    '<stop offset="0%" stop-color="#FFF8EE"/>'
    '<stop offset="100%" stop-color="#F5DDB8"/>'
    '</radialGradient>'
    '<radialGradient id="dg_body" cx="40%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#FAFAFA"/>'
    '<stop offset="100%" stop-color="#E2E2E2"/>'
    '</radialGradient>'
    '</defs>'
    '<ellipse cx="60" cy="114" rx="42" ry="52" fill="url(#dg_body)" stroke="#C8C8C8" stroke-width="1.2"/>'
    '<path d="M44 102 L60 120 L76 102 L76 160 Q60 166 44 160 Z" fill="white" stroke="#E0E0E0" stroke-width="0.8"/>'
    '<path d="M52 102 L60 114" stroke="#CCC" stroke-width="0.8" fill="none"/>'
    '<path d="M68 102 L60 114" stroke="#CCC" stroke-width="0.8" fill="none"/>'
    '<circle cx="60" cy="60" r="40" fill="url(#dg_face)" stroke="#E0C898" stroke-width="1.5"/>'
    '<path d="M24 52 Q26 28 60 24 Q94 28 96 52" fill="#5D4037" opacity="0.5"/>'
    '<ellipse cx="44" cy="64" rx="11.5" ry="10" fill="rgba(255,255,255,0.55)" stroke="#5D4037" stroke-width="1.9"/>'
    '<ellipse cx="76" cy="64" rx="11.5" ry="10" fill="rgba(255,255,255,0.55)" stroke="#5D4037" stroke-width="1.9"/>'
    '<line x1="55.5" y1="63" x2="64.5" y2="63" stroke="#5D4037" stroke-width="1.9"/>'
    '<line x1="32.5" y1="61" x2="25" y2="59" stroke="#5D4037" stroke-width="1.5"/>'
    '<line x1="87.5" y1="61" x2="95" y2="59" stroke="#5D4037" stroke-width="1.5"/>'
    '<circle cx="44" cy="65" r="6" fill="#2C1B35"/>'
    '<circle cx="76" cy="65" r="6" fill="#2C1B35"/>'
    '<circle cx="46.5" cy="62" r="2.2" fill="white" opacity="0.88"/>'
    '<circle cx="78.5" cy="62" r="2.2" fill="white" opacity="0.88"/>'
    '<circle cx="41.5" cy="67" r="1.0" fill="white" opacity="0.5"/>'
    '<circle cx="73.5" cy="67" r="1.0" fill="white" opacity="0.5"/>'
    '<ellipse cx="31" cy="78" rx="11" ry="7" fill="#FFCCBC" opacity="0.38"/>'
    '<ellipse cx="89" cy="78" rx="11" ry="7" fill="#FFCCBC" opacity="0.38"/>'
    '<ellipse cx="60" cy="77" rx="2.5" ry="1.8" fill="#C4956A" opacity="0.5"/>'
    '<path d="M48 88 Q60 100 72 88" stroke="#A06040" stroke-width="2.3" fill="none" stroke-linecap="round"/>'
    '<path d="M44 114 Q36 126 38 140 Q40 150 50 150 Q58 150 58 144" stroke="#78909C" stroke-width="2.8" fill="none" stroke-linecap="round"/>'
    '<circle cx="58" cy="146" r="7" fill="#546E7A" stroke="#263238" stroke-width="1.5"/>'
    '<circle cx="58" cy="146" r="3.5" fill="#37474F"/>'
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
