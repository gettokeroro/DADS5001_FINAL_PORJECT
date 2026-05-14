"""
Global styling injector — Phase 6 + Phase 8 polish
===================================================

inject_global_css()    — BG_R1 image · sidebar gradient · navy text (ทุก page)
inject_ai_mode_css()   — Pastel AI Mode theme (2_AI_Mode.py เท่านั้น · เรียกต่อจาก inject_global_css)
render_nurse_mascot()  — Nurse mascot box + speech (AI Mode Q-screens)
render_doctor_mascot() — Doctor mascot box + speech (AI Mode diagnosis screen)

วิธีใช้:
    from utils.styling import inject_global_css, inject_ai_mode_css, render_nurse_mascot, render_doctor_mascot
    inject_global_css()     # ทุก page
    inject_ai_mode_css()    # เฉพาะ 2_AI_Mode.py
"""
from __future__ import annotations
from pathlib import Path
import base64
import streamlit as st


# ---------------------------------------------------------------------------
# Palette (ตรงกับ .streamlit/config.toml)
# ---------------------------------------------------------------------------
NAVY = "#02497E"          # textColor หลัก
ACCENT = "#22B7E9"        # primaryColor (link, button)
PALE = "#EEFEFF"          # secondaryBackgroundColor (sidebar)
PAGE_BG = "#F8FAFC"       # backgroundColor
BORDER = "#9DE0E9"        # borders

# ---------------------------------------------------------------------------
# AI Mode palette (pastel · Phase 8 mockup v4 approved)
# ---------------------------------------------------------------------------
AI_BODY_TEXT    = "#042C53"   # navy 900 · body text
AI_SUB_TEXT     = "#185FA5"   # navy 600 · sub-labels
AI_BORDER       = "#85B7EB"   # pastel blue border
AI_SCREEN_BG    = "#F4F8FC"   # light blue tint (question screens)
AI_DIAG_BG      = "#FDF4F8"   # light pink (diagnosis screen)
AI_HOVER_BG     = "#D4EDDA"   # pastel green hover
AI_HOVER_TXT    = "#1B5E20"   # dark green text on hover
AI_DIAGNOSE_BTN = "#3DAA6D"   # "ขอคำวินิจฉัย" green button
AI_NARRATION_BG = "rgba(255,255,255,0.85)"
AI_NARRATION_TXT = "#4B1528"  # maroon on light pink

# Sidebar gradient (5 stops ฟ้าอ่อน → ฟ้ากลาง · เลี่ยง navy เข้มล่างสุด)
SIDEBAR_GRADIENT = (
    "linear-gradient(180deg, "
    "#EEFEFF 0%, "
    "#9DE0E9 30%, "
    "#22B7E9 60%, "
    "#3392D2 85%, "
    "#3891CE 100%)"
)

# Overlay opacity บน BG image (สูง = อ่านชัด · ต่ำ = เห็น BG ชัดกว่า)
OVERLAY_ALPHA = 0.78


# ---------------------------------------------------------------------------
# Background image loader (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _bg_data_url() -> str:
    """อ่าน BG_R1_web.jpg แล้ว encode เป็น data URL · cached เพื่อให้ load
    ครั้งเดียวต่อ session"""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "BG_R1_web.jpg",   # compressed (ขนาดเล็ก)
        project_root / "BG_R1.png",        # fallback ถ้ายังไม่มี compressed
    ]
    for p in candidates:
        if p.exists():
            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    return ""


# ---------------------------------------------------------------------------
# CSS injector
# ---------------------------------------------------------------------------
def inject_global_css():
    """Inject CSS ครบชุด: BG image · sidebar gradient · navy text override.

    ต้องเรียกหลัง st.set_page_config() ในทุก page เพราะ Streamlit
    multipage จะ reload CSS ใหม่เมื่อสลับหน้า
    """
    bg = _bg_data_url()
    bg_layer = (
        f'linear-gradient(rgba(248,250,252,{OVERLAY_ALPHA}), '
        f'rgba(248,250,252,{OVERLAY_ALPHA})), '
        f'url("{bg}")'
    ) if bg else "none"

    css = f"""
    <style>
    /* ============================================================
       Main content area — BG_R1 image + white overlay 78%
       ใช้ background-attachment: fixed → BG อยู่กับที่เหมือน wallpaper
       ============================================================ */
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] > .main {{
        background-image: {bg_layer};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* ============================================================
       Sidebar — gradient ฟ้าอ่อนบน → ฟ้ากลางล่าง
       ============================================================ */
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_GRADIENT};
    }}
    [data-testid="stSidebar"] *:not(button):not(a) {{
        color: {NAVY} !important;
    }}
    /* sidebar nav links — กล่องขาวจางๆ ให้อ่านง่ายบน gradient */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        background: rgba(255,255,255,0.45);
        border-radius: 6px;
        margin-bottom: 4px;
        padding: 6px 10px;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        background: rgba(255,255,255,0.75);
    }}
    /* active page (current) */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: rgba(255,255,255,0.92);
        font-weight: 500;
    }}

    /* ============================================================
       Main text — navy ทั้งหมด · captions ใช้ opacity 0.72
       ============================================================ */
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{
        color: {NAVY};
    }}
    /* captions = navy with opacity (ลำดับชั้น) */
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] small,
    [data-testid="stMain"] .stCaption {{
        color: {NAVY} !important;
        opacity: 0.72;
    }}
    /* links = ฟ้าสด · hover = navy */
    [data-testid="stMain"] a {{
        color: {ACCENT};
        text-decoration: none;
    }}
    [data-testid="stMain"] a:hover {{
        color: {NAVY};
        text-decoration: underline;
    }}

    /* ============================================================
       Cards / containers — พื้นขาว 92% เพื่อให้ text ชัดบน BG
       ============================================================ */
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(2px);
        border-radius: 8px;
    }}
    /* expanders */
    [data-testid="stMain"] [data-testid="stExpander"] > details {{
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(2px);
        border-radius: 8px;
        border: 0.5px solid {BORDER};
    }}

    /* ============================================================
       Header bar (Streamlit's top toolbar) — โปร่งใสเพื่อเห็น BG
       ============================================================ */
    [data-testid="stHeader"] {{
        background: rgba(248, 250, 252, 0.6);
        backdrop-filter: blur(4px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AI Mode CSS injector (Phase 8 · เรียกหลัง inject_global_css)
# ---------------------------------------------------------------------------
def inject_ai_mode_css() -> None:
    """Inject CSS เพิ่มเติมสำหรับ AI Mode pastel theme.

    ต้องเรียกหลัง inject_global_css() ใน pages/2_AI_Mode.py เท่านั้น
    เพื่อ override global navy theme ให้เป็น pastel สำหรับหน้านี้โดยเฉพาะ
    """
    css = f"""
    <style>
    /* ================================================================
       AI Mode — Pastel theme (Phase 8 · mockup v4 approved)
       ================================================================ */

    /* Question screen container */
    .ai-screen {{
        background: {AI_SCREEN_BG};
        border: 1.5px solid {AI_BORDER};
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 14px;
    }}
    .ai-screen.diagnosis {{
        background: {AI_DIAG_BG};
    }}

    /* Body text override — always dark on pastel bg */
    .ai-screen p,
    .ai-screen span,
    .ai-screen label,
    .ai-screen li,
    .ai-screen h3,
    .ai-screen h4 {{
        color: {AI_BODY_TEXT} !important;
        font-weight: 500;
    }}
    .ai-screen .ai-sub {{
        color: {AI_SUB_TEXT} !important;
        font-size: 13px;
        font-weight: 400;
    }}

    /* Mascot box (nurse / doctor) */
    .ai-mascot-box {{
        display: flex;
        align-items: center;
        gap: 16px;
        background: rgba(255,255,255,0.72);
        border: 1.5px solid {AI_BORDER};
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 18px;
    }}
    .ai-mascot-box svg {{
        flex-shrink: 0;
    }}
    .ai-mascot-speech {{
        color: {AI_BODY_TEXT} !important;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.55;
    }}
    .ai-mascot-speech .ai-sub {{
        display: block;
        font-weight: 400;
        font-size: 13px;
        color: {AI_SUB_TEXT} !important;
        margin-top: 4px;
    }}

    /* Symptom basket tags */
    .ai-tag {{
        display: inline-block;
        background: #E3F2FD;
        border: 1px solid {AI_BORDER};
        border-radius: 20px;
        padding: 3px 11px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        color: {AI_BODY_TEXT} !important;
        font-weight: 500;
    }}

    /* Progress step dots */
    .ai-progress {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }}
    .ai-dot {{
        width: 9px; height: 9px;
        border-radius: 50%;
        background: {AI_BORDER};
    }}
    .ai-dot.done  {{ background: #66BB6A; }}
    .ai-dot.now   {{ background: {AI_SUB_TEXT}; transform: scale(1.4); }}

    /* Doctor narration text box */
    .ai-narration {{
        background: {AI_NARRATION_BG};
        border-left: 4px solid #F48FB1;
        border-radius: 0 8px 8px 0;
        padding: 13px 17px;
        color: {AI_NARRATION_TXT} !important;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.6;
        margin-bottom: 14px;
    }}

    /* Streamlit buttons in AI Mode — pastel green hover */
    [data-testid="stMain"] .stButton > button {{
        background: white !important;
        border: 1.5px solid {AI_BORDER} !important;
        border-radius: 8px !important;
        color: {AI_BODY_TEXT} !important;
        font-weight: 500 !important;
        transition: background 0.15s, border-color 0.15s, color 0.15s;
    }}
    [data-testid="stMain"] .stButton > button:hover {{
        background: {AI_HOVER_BG} !important;
        border-color: #81C784 !important;
        color: {AI_HOVER_TXT} !important;
    }}
    [data-testid="stMain"] .stButton > button:active {{
        background: #C8E6C9 !important;
    }}

    /* "ขอคำวินิจฉัย" green button — ใช้ key class diagnose-btn ผ่าน st.markdown */
    .diagnose-btn button,
    [data-testid="stMain"] .stButton.diagnose-btn > button {{
        background: {AI_DIAGNOSE_BTN} !important;
        border-color: {AI_DIAGNOSE_BTN} !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }}
    [data-testid="stMain"] .stButton.diagnose-btn > button:hover {{
        background: #2E8B57 !important;
        border-color: #2E8B57 !important;
        color: white !important;
    }}

    /* Free-text input */
    [data-testid="stMain"] .stTextInput input {{
        background: white !important;
        border: 1.5px solid {AI_BORDER} !important;
        border-radius: 8px !important;
        color: {AI_BODY_TEXT} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Mascot SVG loader + renderer
# ---------------------------------------------------------------------------
def _load_mascot_svg(name: str) -> str:
    """โหลด SVG จาก assets/mascot_{name}.svg · fallback เป็น emoji placeholder"""
    p = Path(__file__).resolve().parent.parent / "assets" / f"mascot_{name}.svg"
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Emoji fallback (กรณี SVG ไม่มี)
    emoji = "🩺" if name == "doctor" else "💊"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60" width="70" height="70">'
        f'<circle cx="30" cy="30" r="28" fill="#E0E0E0"/>'
        f'<text x="30" y="38" text-anchor="middle" font-size="28">{emoji}</text>'
        f'</svg>'
    )


def render_nurse_mascot(speech: str, sub: str = "") -> None:
    """แสดงกล่อง nurse mascot พร้อม speech bubble (ใช้ใน Q1-Q3 screens)"""
    svg = _load_mascot_svg("nurse")
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        f"""<div class="ai-mascot-box">
              <div style="width:88px;flex-shrink:0">{svg}</div>
              <div class="ai-mascot-speech">{speech}{sub_html}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_doctor_mascot(speech: str, sub: str = "") -> None:
    """แสดงกล่อง doctor mascot พร้อม speech bubble (ใช้ใน diagnosis screen)"""
    svg = _load_mascot_svg("doctor")
    sub_html = f'<span class="ai-sub">{sub}</span>' if sub else ""
    st.markdown(
        f"""<div class="ai-mascot-box">
              <div style="width:88px;flex-shrink:0">{svg}</div>
              <div class="ai-mascot-speech">{speech}{sub_html}</div>
            </div>""",
        unsafe_allow_html=True,
    )
