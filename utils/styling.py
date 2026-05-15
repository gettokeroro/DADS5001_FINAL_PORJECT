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
        f".ai-mascot-speech{{color:{AI_BODY_TEXT} !important;font-weight:600;line-height:1.55;}}",
        f".ai-mascot-speech .ai-sub{{display:block;font-weight:400;font-size:13px;color:{AI_SUB_TEXT} !important;margin-top:4px;}}",
        # v2 vertical card layout (mascot ด้านบน · ชื่อ · คำพูด)
        ".ai-mascot-vcard{display:flex;flex-direction:column;align-items:center;text-align:center;gap:8px;background:rgba(255,248,251,0.85);border:1.5px solid #F4C0D1;border-radius:14px;padding:16px 20px;margin:0 auto 18px;max-width:420px;}",
        ".ai-mascot-vcard.doctor{background:rgba(255,252,232,0.85);border-color:#F8D898;}",
        ".ai-mascot-vcard svg{display:block;width:130px;height:auto;}",
        f".ai-mascot-vcard .vc-name{{font-size:15px;font-weight:700;color:#4B1528;letter-spacing:0.2px;}}",
        ".ai-mascot-vcard.doctor .vc-name{color:#6B5028;}",
        f".ai-mascot-vcard .vc-speech{{font-size:14px;font-weight:500;line-height:1.6;color:#4B1528;max-width:340px;}}",
        ".ai-mascot-vcard.doctor .vc-speech{color:#6B5028;}",
        f".ai-mascot-vcard .ai-sub{{display:block;font-size:12px;font-weight:400;font-style:italic;color:{AI_SUB_TEXT} !important;margin-top:6px;}}",
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
    # v2 (2026-05-15): Chansey-true — egg-shape blob เดียว ไม่แยก head/body
    # · hair curls 3 จุดแบบ chansey signature · พุงขาว · flipper arms · หมวก + heart pill
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 170" width="140" height="170">'
    '<defs>'
    '<radialGradient id="nv2_body" cx="38%" cy="28%" r="78%">'
    '<stop offset="0%" stop-color="#FFF0F6"/>'
    '<stop offset="55%" stop-color="#FFC8DD"/>'
    '<stop offset="100%" stop-color="#FF8FB5"/>'
    '</radialGradient>'
    '<radialGradient id="nv2_iris" cx="30%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#7C3996"/>'
    '<stop offset="100%" stop-color="#2C1050"/>'
    '</radialGradient>'
    '</defs>'
    # Unified egg body
    '<ellipse cx="70" cy="92" rx="56" ry="66" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="2"/>'
    # White belly patch
    '<ellipse cx="70" cy="125" rx="34" ry="25" fill="white" opacity="0.75" stroke="#FFAAC4" stroke-width="0.6"/>'
    # Stub flipper arms
    '<ellipse cx="18" cy="98" rx="9" ry="15" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="1.5" transform="rotate(-12 18 98)"/>'
    '<ellipse cx="122" cy="98" rx="9" ry="15" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="1.5" transform="rotate(12 122 98)"/>'
    # Hair curls (chansey signature, 3 tufts)
    '<path d="M 48 30 Q 44 18 52 20 Q 54 14 58 22" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="1.4" stroke-linejoin="round"/>'
    '<path d="M 64 28 Q 60 14 68 16 Q 72 8 76 18 Q 80 14 78 26" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="1.4" stroke-linejoin="round"/>'
    '<path d="M 82 30 Q 84 18 92 22 Q 96 16 96 28" fill="url(#nv2_body)" stroke="#FF80A8" stroke-width="1.4" stroke-linejoin="round"/>'
    # Nurse cap
    '<ellipse cx="70" cy="20" rx="26" ry="6" fill="white" stroke="#E0D0DB" stroke-width="0.8"/>'
    '<rect x="55" y="12" width="30" height="11" rx="4" fill="white" stroke="#E0D0DB" stroke-width="0.8"/>'
    '<rect x="66" y="14" width="8" height="2.5" rx="0.5" fill="#E53935"/>'
    '<rect x="68.5" y="11.5" width="3" height="7.5" rx="0.5" fill="#E53935"/>'
    # Eyes
    '<ellipse cx="48" cy="78" rx="6.5" ry="9" fill="url(#nv2_iris)"/>'
    '<ellipse cx="92" cy="78" rx="6.5" ry="9" fill="url(#nv2_iris)"/>'
    '<circle cx="48" cy="80" r="4" fill="#0C0418"/>'
    '<circle cx="92" cy="80" r="4" fill="#0C0418"/>'
    '<circle cx="51" cy="74" r="2.8" fill="white"/>'
    '<circle cx="95" cy="74" r="2.8" fill="white"/>'
    '<circle cx="45" cy="82" r="1.2" fill="white" opacity="0.7"/>'
    '<circle cx="89" cy="82" r="1.2" fill="white" opacity="0.7"/>'
    # Eyebrows
    '<path d="M 40 64 Q 48 60 56 64" stroke="#7C3996" stroke-width="1.6" fill="none" stroke-linecap="round" opacity="0.6"/>'
    '<path d="M 84 64 Q 92 60 100 64" stroke="#7C3996" stroke-width="1.6" fill="none" stroke-linecap="round" opacity="0.6"/>'
    # Blush
    '<ellipse cx="28" cy="95" rx="10" ry="6" fill="#FF6090" opacity="0.35"/>'
    '<ellipse cx="112" cy="95" rx="10" ry="6" fill="#FF6090" opacity="0.35"/>'
    # Mouth
    '<path d="M 58 100 Q 70 110 82 100" stroke="#E03060" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    '<ellipse cx="70" cy="105" rx="6" ry="3" fill="#FFAAC4" opacity="0.5"/>'
    # Heart pill on belly
    '<path d="M 70 138 C 70 138 62 130 62 124 Q 62 119 67 119 Q 70 120 70 123 Q 70 120 73 119 Q 78 119 78 124 C 78 130 70 138 70 138 Z" fill="#E53935"/>'
    '<path d="M 65 124 Q 66 121 68 121" stroke="white" stroke-width="1.2" fill="none" opacity="0.7"/>'
    '</svg>'
)

_DOCTOR_SVG = (
    # v2 (2026-05-15): เหลืองพาสเทล egg-shape · แว่นกลมใหญ่ · stethoscope พาดตัว · กาวน์ขาว V-neck
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 170" width="140" height="170">'
    '<defs>'
    '<radialGradient id="dv2_body" cx="38%" cy="28%" r="78%">'
    '<stop offset="0%" stop-color="#FFFCEF"/>'
    '<stop offset="55%" stop-color="#FFE9A8"/>'
    '<stop offset="100%" stop-color="#F4CD6B"/>'
    '</radialGradient>'
    '<radialGradient id="dv2_iris" cx="30%" cy="28%" r="72%">'
    '<stop offset="0%" stop-color="#3478A3"/>'
    '<stop offset="100%" stop-color="#0A2540"/>'
    '</radialGradient>'
    '</defs>'
    # Unified yellow-pastel egg body
    '<ellipse cx="70" cy="92" rx="56" ry="66" fill="url(#dv2_body)" stroke="#E8B66E" stroke-width="2"/>'
    # White doctor coat lapel
    '<path d="M30 120 Q50 158 70 158 Q90 158 110 120 L105 156 L35 156 Z" fill="white" opacity="0.85" stroke="#E0E0E0" stroke-width="0.7"/>'
    '<line x1="55" y1="130" x2="70" y2="148" stroke="#D0D0D0" stroke-width="0.8"/>'
    '<line x1="85" y1="130" x2="70" y2="148" stroke="#D0D0D0" stroke-width="0.8"/>'
    # Buttons
    '<circle cx="70" cy="135" r="1.3" fill="#B0B0B0"/>'
    '<circle cx="70" cy="143" r="1.3" fill="#B0B0B0"/>'
    # Stub flipper arms
    '<ellipse cx="18" cy="98" rx="9" ry="15" fill="url(#dv2_body)" stroke="#E8B66E" stroke-width="1.5" transform="rotate(-12 18 98)"/>'
    '<ellipse cx="122" cy="98" rx="9" ry="15" fill="url(#dv2_body)" stroke="#E8B66E" stroke-width="1.5" transform="rotate(12 122 98)"/>'
    # Brown hair tuft on top
    '<path d="M60 28 Q56 14 66 18 Q70 8 76 18 Q84 14 80 30" fill="#8D6E63" stroke="#5D4037" stroke-width="1" stroke-linejoin="round"/>'
    '<path d="M64 22 Q70 14 76 22" stroke="#5D4037" stroke-width="0.8" fill="none" opacity="0.4"/>'
    # Round glasses
    '<circle cx="48" cy="80" r="15" fill="rgba(255,255,255,0.55)" stroke="#5D4037" stroke-width="2.6"/>'
    '<circle cx="92" cy="80" r="15" fill="rgba(255,255,255,0.55)" stroke="#5D4037" stroke-width="2.6"/>'
    '<path d="M63 80 Q70 76 77 80" stroke="#5D4037" stroke-width="2.4" fill="none"/>'
    # Glasses temples
    '<line x1="33" y1="76" x2="22" y2="74" stroke="#5D4037" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="107" y1="76" x2="118" y2="74" stroke="#5D4037" stroke-width="1.8" stroke-linecap="round"/>'
    # Eyes inside glasses
    '<ellipse cx="48" cy="82" rx="5" ry="6.5" fill="url(#dv2_iris)"/>'
    '<ellipse cx="92" cy="82" rx="5" ry="6.5" fill="url(#dv2_iris)"/>'
    '<circle cx="48" cy="83" r="2.8" fill="#050E20"/>'
    '<circle cx="92" cy="83" r="2.8" fill="#050E20"/>'
    '<circle cx="51" cy="78" r="2.2" fill="white"/>'
    '<circle cx="95" cy="78" r="2.2" fill="white"/>'
    # Blush
    '<ellipse cx="26" cy="102" rx="10" ry="6" fill="#FFAA88" opacity="0.32"/>'
    '<ellipse cx="114" cy="102" rx="10" ry="6" fill="#FFAA88" opacity="0.32"/>'
    # Gentle smile
    '<path d="M58 104 Q70 113 82 104" stroke="#A06040" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    '<ellipse cx="70" cy="108" rx="6" ry="2.5" fill="#E8B66E" opacity="0.45"/>'
    # Stethoscope
    '<path d="M50 118 Q30 124 28 140 Q26 154 38 154" stroke="#37474F" stroke-width="2.4" fill="none"/>'
    '<path d="M90 118 Q110 124 112 140" stroke="#37474F" stroke-width="2.4" fill="none"/>'
    '<circle cx="38" cy="154" r="6.5" fill="#546E7A" stroke="#263238" stroke-width="1.3"/>'
    '<circle cx="38" cy="154" r="3.2" fill="#37474F"/>'
    '<circle cx="38" cy="154" r="1.2" fill="#B0BEC5"/>'
    '</svg>'
)


def render_nurse_mascot(speech: str, sub: str = "", name: str = "น้องอุ่นใน 💕") -> None:
    """Vertical card layout — SVG ด้านบน · ชื่อ · คำพูดด้านล่าง"""
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        '<div class="ai-mascot-vcard">' +
        _NURSE_SVG +
        f'<div class="vc-name">{name}</div>' +
        f'<div class="vc-speech">{speech}{sub_html}</div>' +
        '</div>',
        unsafe_allow_html=True,
    )


def render_doctor_mascot(speech: str, sub: str = "", name: str = "หมอใจดี 🩺") -> None:
    """Vertical card layout — SVG ด้านบน · ชื่อ · คำพูดด้านล่าง"""
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        '<div class="ai-mascot-vcard doctor">' +
        _DOCTOR_SVG +
        f'<div class="vc-name">{name}</div>' +
        f'<div class="vc-speech">{speech}{sub_html}</div>' +
        '</div>',
        unsafe_allow_html=True,
    )
