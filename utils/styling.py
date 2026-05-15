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
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="88" height="124">' +
    '<ellipse cx="60" cy="96" rx="46" ry="62" style="fill:#FBC8D8;stroke:#EC9AB0;stroke-width:1.5"/>' +
    '<rect x="28" y="28" width="64" height="10" rx="5" style="fill:white;stroke:#D0D0D0;stroke-width:1"/>' +
    '<rect x="33" y="16" width="54" height="18" rx="5" style="fill:white;stroke:#D0D0D0;stroke-width:1"/>' +
    '<rect x="55" y="19" width="10" height="3.5" rx="1.5" style="fill:#E53935"/>' +
    '<rect x="58.5" y="15.5" width="3.5" height="10" rx="1.5" style="fill:#E53935"/>' +
    '<circle cx="45" cy="76" r="6" style="fill:#2D3748"/>' +
    '<circle cx="75" cy="76" r="6" style="fill:#2D3748"/>' +
    '<circle cx="47.5" cy="73.5" r="2" style="fill:white"/>' +
    '<circle cx="77.5" cy="73.5" r="2" style="fill:white"/>' +
    '<ellipse cx="35" cy="88" rx="10" ry="6" style="fill:#F48FB1;opacity:0.42"/>' +
    '<ellipse cx="85" cy="88" rx="10" ry="6" style="fill:#F48FB1;opacity:0.42"/>' +
    '<path d="M47 96 Q60 110 73 96" style="stroke:#2D3748;stroke-width:2.5;fill:none;stroke-linecap:round"/>' +
    '<path d="M42 120 L60 134 L78 120" style="fill:white;stroke:#DDD;stroke-width:1.2"/>' +
    '<path d="M60 148 C60 148 48 140 48 132 Q48 126 54 126 Q57 126 60 130 Q63 126 66 126 Q72 126 72 132 C72 140 60 148 60 148 Z" style="fill:#E53935"/>' +
    '</svg>'
)

_DOCTOR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 168" width="88" height="124">' +
    '<ellipse cx="60" cy="96" rx="46" ry="62" style="fill:#FFF8E7;stroke:#E8D080;stroke-width:1.5"/>' +
    '<path d="M42 118 L60 136 L78 118 L78 158 Q60 164 42 158 Z" style="fill:white;stroke:#E0E0E0;stroke-width:1"/>' +
    '<path d="M52 118 L60 128" style="stroke:#CCC;stroke-width:1;fill:none"/>' +
    '<path d="M68 118 L60 128" style="stroke:#CCC;stroke-width:1;fill:none"/>' +
    '<circle cx="45" cy="76" r="6" style="fill:#2D3748"/>' +
    '<circle cx="75" cy="76" r="6" style="fill:#2D3748"/>' +
    '<circle cx="47.5" cy="73.5" r="2" style="fill:white"/>' +
    '<circle cx="77.5" cy="73.5" r="2" style="fill:white"/>' +
    '<ellipse cx="45" cy="76" rx="10.5" ry="8.5" style="fill:none;stroke:#5D4037;stroke-width:1.8"/>' +
    '<ellipse cx="75" cy="76" rx="10.5" ry="8.5" style="fill:none;stroke:#5D4037;stroke-width:1.8"/>' +
    '<line x1="55.5" y1="76" x2="64.5" y2="76" style="stroke:#5D4037;stroke-width:1.8"/>' +
    '<line x1="34.5" y1="73" x2="27" y2="71" style="stroke:#5D4037;stroke-width:1.5"/>' +
    '<line x1="85.5" y1="73" x2="93" y2="71" style="stroke:#5D4037;stroke-width:1.5"/>' +
    '<ellipse cx="35" cy="88" rx="10" ry="6" style="fill:#FFCCBC;opacity:0.45"/>' +
    '<ellipse cx="85" cy="88" rx="10" ry="6" style="fill:#FFCCBC;opacity:0.45"/>' +
    '<path d="M47 96 Q60 110 73 96" style="stroke:#2D3748;stroke-width:2.5;fill:none;stroke-linecap:round"/>' +
    '<path d="M44 114 Q36 126 38 138 Q40 148 50 148 Q58 148 58 140" style="stroke:#78909C;stroke-width:2.8;fill:none;stroke-linecap:round"/>' +
    '<circle cx="58" cy="142" r="7" style="fill:#546E7A;stroke:#263238;stroke-width:1.5"/>' +
    '<circle cx="58" cy="142" r="3.5" style="fill:#37474F"/>' +
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
