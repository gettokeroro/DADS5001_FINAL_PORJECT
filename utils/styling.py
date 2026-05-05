"""
Global styling injector — Phase 6 polish
========================================

ใส่ BG_R1 image พร้อม overlay 78%, gradient sidebar (ฟ้าอ่อน → ฟ้ากลาง),
และ override ตัวอักษรในพื้นที่ขวา (main) ให้เป็นสี navy #02497E ทั้งหมด

วิธีใช้:
    from utils.styling import inject_global_css
    inject_global_css()      # call หลัง st.set_page_config() ในทุก page
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
