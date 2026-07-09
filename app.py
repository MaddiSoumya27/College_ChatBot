"""
app.py
------
Phase 4: Streamlit Chat UI for BVRIT Hyderabad RAG Chatbot.
Updated: Wired with all five memory layers from memory.py.

Run:  streamlit run app.py
"""

import os
import json
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Ensure working directory is the script's own folder ──────────────────────
# This fixes issues when Streamlit is launched from a different directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import run_rag_pipeline, get_vectorstore, reset_vectorstore

# ── Memory layers ─────────────────────────────────────────────────────────────
from memory import (
    # Layer 1 — Short-term
    make_user_turn, make_assistant_turn, get_llm_history,
    # Layer 2 — Medium-term
    maybe_summarize, get_llm_history_with_summary,
    # Layer 3 — Long-term
    init_db, load_profile, save_profile, extract_and_update_profile,
    # Layer 4 — Personalization
    build_personalization_prefix,
    # Layer 5 — Privacy
    expire_old_profiles, is_clear_data_command, handle_clear_data,
    get_or_create_user_id, PRIVACY_NOTICE,
)

# ── One-time startup tasks ────────────────────────────────────────────────────
init_db()           # ensure SQLite table exists
expire_old_profiles()  # Layer 5: remove profiles idle > 30 days

# ── Image gallery — keyword → list of (caption, url) ─────────────────────────
# All URLs verified from bvrithyderabad.edu.in official site
BASE = "https://bvrithyderabad.edu.in/wp-content/uploads"
COLLEGE_IMAGES: dict[str, list[tuple[str, str]]] = {
    "campus": [
        ("Welcome to BVRIT Hyderabad", f"{BASE}/2026/01/WelcomeBVRITH.jpg"),
        ("APJ Block — Main Entrance", f"{BASE}/2026/01/ApjIngate.jpg"),
        ("SMB Block", f"{BASE}/2026/01/SMB1.jpg"),
        ("CSE — Sita Block", f"{BASE}/2026/01/CSE-SITA-BLOCK1.jpg"),
        ("Memorial Hall", f"{BASE}/2026/01/MemorialHall1.jpg"),
        ("Home slider — Campus view", f"{BASE}/2023/04/home-slider-6-bvrit-hyderabad-engineering-women-college.webp"),
    ],
    "lab": [
        ("Drone Technology Laboratory", f"{BASE}/2023/06/DroneBannerImage.jpg"),
        ("DTL — Top view", f"{BASE}/2023/06/DTL-top-image-1.jpeg"),
        ("DTL — Scrolling view", f"{BASE}/2023/06/Scrolling-image-1-DTL.jpeg"),
    ],
    "placement": [
        ("Google Placements", f"{BASE}/2026/02/GooglePlacements.jpeg"),
        ("Accenture Placements", f"{BASE}/2023/04/accenture-placement-bvrit-hyderabad-engineering-women-college.webp"),
        ("Amazon Placements", f"{BASE}/2023/04/amazon-placement-bvrit-hyderabad-engineering-women-college.webp"),
        ("Flipkart Placements", f"{BASE}/2023/04/flipkart-placement-bvrit-hyderabad-engineering-women-college.webp"),
        ("Infosys Placements", f"{BASE}/2023/04/infosys-placement-bvrit-hyderabad-engineering-women-college.webp"),
    ],
    "event": [
        ("Synergia 2026 — Tech & Cultural Fest", f"{BASE}/2026/04/Synergia-26.jpg"),
        ("Annual Day 2026", f"{BASE}/2026/04/annualday1.jpg"),
        ("Graduation Day 2025", f"{BASE}/2025/11/Graduation-Day-2025.jpg"),
        ("TedX at BVRIT Hyderabad", f"{BASE}/2026/02/TedX.jpg"),
        ("Milan 2026", f"{BASE}/2026/03/Milan2026.jpg"),
    ],
    "research": [
        ("IIC — Innovation Council", f"{BASE}/2023/06/IIC-1024x682-1.jpg"),
        ("EDC — Entrepreneurship Cell", f"{BASE}/2023/06/EDCImage1-1024x768-1.jpg"),
        ("SIH 2025 Winners", f"{BASE}/2025/12/SIH-2025-winners-1.jpg"),
    ],
    "nss": [
        ("NSS — First Aid Training", f"{BASE}/2023/06/NSS-FirstAidTraining.jpg"),
        ("NSS — Go Green Rally", f"{BASE}/2023/06/NSS-GoGreenRally.jpg"),
    ],
    "principal": [
        ("Dr. K.V.N. Sunitha — Principal Award", f"{BASE}/2021/09/Principal-Award-copy.jpg"),
        ("CSI Award — Best Supporting Principal", f"{BASE}/2025/06/csiAward.jpg"),
    ],
    "management": [
        ("Chairman — Sri K.V. Vishnu Raju", f"{BASE}/2025/09/Chairman-Sir.jpg"),
        ("Vice Chairman", f"{BASE}/2025/09/VC-Sir.jpg"),
    ],
    "yoga": [
        ("Yoga at BVRIT Hyderabad", f"{BASE}/2023/05/yoga-7-bvrith-image-bvrit-hyderabad-engineering-women-college.webp"),
    ],
    "general": [
        ("Welcome to BVRIT Hyderabad", f"{BASE}/2026/01/WelcomeBVRITH.jpg"),
        ("APJ Block — Main Entrance", f"{BASE}/2026/01/ApjIngate.jpg"),
        ("Synergia 2026 Fest", f"{BASE}/2026/04/Synergia-26.jpg"),
        ("Home section", f"{BASE}/2023/04/home-section-img2-bvrit-hyderabad-engineering-women-college.webp"),
    ],
}

# Keywords that map user query words to image categories
IMAGE_KEYWORD_MAP = {
    "campus":     ["campus", "college", "building", "outside", "front", "entrance", "block", "infrastructure"],
    "lab":        ["lab", "laboratory", "drone", "iot", "maker space", "practical", "dtl"],
    "placement":  ["placement", "tap cell", "recruitment", "recruiter", "job", "accenture", "amazon", "google", "flipkart"],
    "event":      ["event", "fest", "synergia", "annual day", "graduation", "tedx", "milan", "cultural"],
    "research":   ["research", "innovation", "iic", "edc", "sih", "hackathon", "startup"],
    "nss":        ["nss", "community", "social", "outreach", "rally", "volunteer"],
    "principal":  ["principal", "dr sunitha", "kvn", "hod", "director"],
    "management": ["chairman", "management", "vice chairman", "secretary", "leadership"],
    "yoga":       ["yoga", "fitness", "wellness", "sports", "gym"],
}

IMAGE_REQUEST_KEYWORDS = [
    "show", "image", "photo", "picture", "pic", "look like",
    "what does", "how does", "view", "see", "gallery", "visual", "photos"
]

IMAGE_ONLY_RESPONSE = (
    "Here are some photos from BVRIT Hyderabad College. "
    "You can find more on the official website at "
    "[bvrithyderabad.edu.in](https://bvrithyderabad.edu.in/gallery/)."
)


def detect_image_request(text: str) -> list[tuple[str, str]]:
    """
    If the user is asking for images, return matching (caption, url) pairs.
    Returns empty list if not an image request.
    """
    lower = text.lower()
    if not any(kw in lower for kw in IMAGE_REQUEST_KEYWORDS):
        return []

    matched_images = []
    for category, triggers in IMAGE_KEYWORD_MAP.items():
        if any(t in lower for t in triggers):
            matched_images.extend(COLLEGE_IMAGES.get(category, []))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in matched_images:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)

    return unique if unique else COLLEGE_IMAGES["general"]


def is_image_only_request(text: str) -> bool:
    """
    Returns True if the query is purely asking for images with no
    informational content needed (e.g. 'show me campus photos').
    """
    lower = text.lower()
    has_image_kw = any(kw in lower for kw in IMAGE_REQUEST_KEYWORDS)
    # If it also has an informational word, let the RAG pipeline handle it too
    info_keywords = ["fee", "admission", "department", "faculty",
                     "contact", "hostel", "transport", "course", "program",
                     "rank", "cutoff", "naac", "nba", "about"]
    has_info_kw = any(kw in lower for kw in info_keywords)
    return has_image_kw and not has_info_kw

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BVRIT Hyderabad — Info Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional UI Theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ════════════════════════════════════════
   GLOBAL
════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background-color: #f4f6fb !important;
}
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }

/* Default text colour for everything in the main area */
.main, .main p, .main li, .main span, .main div,
.main label, .main strong, .main em, .main h1, .main h2, .main h3,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #0a1f44 !important;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f44 0%, #0d2b5e 100%) !important;
    border-right: 1px solid #1a3a6e !important;
}

/* All sidebar text → white by default */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] caption,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #dce8ff !important;
}

/* Sidebar headings → gold */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f5d060 !important;
    font-weight: 700 !important;
}

/* Sidebar labels (sliders, selects, radios) */
[data-testid="stSidebar"] label {
    color: #a8c0f0 !important;
    font-size: 0.82rem !important;
}

/* Sidebar metric */
[data-testid="stSidebar"] [data-testid="stMetricValue"] > div { color: #ffffff !important; font-weight: 700 !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] > div { color: #a8c0f0 !important; }
[data-testid="stSidebar"] [data-testid="stMetricDelta"] > div { color: #4ade80 !important; }

/* Sidebar HR */
[data-testid="stSidebar"] hr { border-color: #1a3a6e !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background-color: #1a3a6e !important;
    color: #ffffff !important;
    border: 1px solid #2e5aad !important;
    border-radius: 6px !important;
    width: 100% !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 0.8rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2e5aad !important;
    color: #ffffff !important;
    border-color: #5080e0 !important;
}

/* Sidebar nav radio labels */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: #1a3a6e !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.8rem !important;
    margin-bottom: 0.3rem !important;
    display: block !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #2e5aad !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #16305a !important;
    border-color: #2e5aad !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #f5d060 !important;
    opacity: 1 !important;
    visibility: visible !important;
    width: 16px !important;
    height: 16px !important;
}

/* Sidebar slider thumb */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #f5d060 !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 0 6px rgba(245,208,96,0.5) !important;
}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: #a8c0f0 !important;
}

/* ════════════════════════════════════════
   HEADER BANNER
════════════════════════════════════════ */
.bvrit-header {
    background: linear-gradient(135deg, #0a1f44 0%, #0d2b5e 60%, #1a3a6e 100%);
    border-bottom: 3px solid #c9a84c;
    padding: 1.1rem 2rem 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(10,31,68,0.18);
}
.bvrit-header-logo { font-size: 2.4rem; line-height: 1; }
.bvrit-header-text h1 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 0.01em;
    line-height: 1.3;
}
.bvrit-header-text p {
    margin: 0.15rem 0 0 0;
    font-size: 0.78rem;
    color: #c9a84c !important;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.bvrit-header-badge {
    margin-left: auto;
    background: rgba(201,168,76,0.15);
    border: 1.5px solid #c9a84c;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.68rem;
    color: #f5d060 !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ════════════════════════════════════════
   MAIN AREA BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background-color: #0a1f44 !important;
    color: #ffffff !important;
    border: 1.5px solid #2e5aad !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    background-color: #2e5aad !important;
    color: #ffffff !important;
    border-color: #6090e8 !important;
}
.stButton > button svg {
    fill: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* ════════════════════════════════════════
   EXPANDER / ACCORDION ARROWS
════════════════════════════════════════ */
[data-testid="stExpander"] summary {
    color: #0a1f44 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg,
details summary svg {
    fill: #0a1f44 !important;
    opacity: 1 !important;
    visibility: visible !important;
    width: 18px !important;
    height: 18px !important;
}
[data-testid="stExpander"] summary:hover svg { fill: #2e5aad !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    color: #0a1f44 !important;
}

/* ════════════════════════════════════════
   SELECTBOX (main area)
════════════════════════════════════════ */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #a0b8e8 !important;
}
[data-baseweb="select"] span {
    color: #0a1f44 !important;
}
[data-baseweb="select"] svg {
    fill: #0a1f44 !important;
    opacity: 1 !important;
    visibility: visible !important;
    width: 16px !important;
    height: 16px !important;
}
/* Dropdown list */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    color: #0a1f44 !important;
    background-color: #ffffff !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background-color: #e8edf8 !important;
}

/* ════════════════════════════════════════
   SIDEBAR COLLAPSE ARROW
════════════════════════════════════════ */
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
button[aria-label*="sidebar"] svg,
button[aria-label*="Sidebar"] svg {
    fill: #0a1f44 !important;
    opacity: 1 !important;
    visibility: visible !important;
    width: 22px !important;
    height: 22px !important;
}

/* ════════════════════════════════════════
   CHAT MESSAGES
════════════════════════════════════════ */
[data-testid="stChatMessageContainer"] { background: transparent; }
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] code {
    color: #0a1f44 !important;
}
[data-testid="stChatMessage"] a { color: #2e5aad !important; text-decoration: underline; }
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #e8edf8;
    border-left: 3px solid #2e5aad;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #f0f4ff;
    border-left: 3px solid #c9a84c;
}

/* ════════════════════════════════════════
   CHAT INPUT
════════════════════════════════════════ */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #0a1f44 !important;
    border: 1.5px solid #a0b8e8 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #6080b0 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #2e5aad !important;
    box-shadow: 0 0 0 3px rgba(46,90,173,0.12) !important;
}
[data-testid="stChatInput"] button {
    background-color: #0a1f44 !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stChatInput"] button svg {
    fill: #f5d060 !important;
    color: #f5d060 !important;
    stroke: #f5d060 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stChatInput"] button:hover { background-color: #2e5aad !important; }

/* ════════════════════════════════════════
   METRICS, ALERTS, DATAFRAME
════════════════════════════════════════ */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    padding: 0.7rem 1rem !important;
    border: 1px solid #e0e6f5 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] > div { color: #6b7a9e !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] > div { color: #0a1f44 !important; font-size: 1.35rem !important; font-weight: 700 !important; }
.stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }
[data-testid="stDataFrame"] { border-radius: 8px !important; border: 1px solid #e0e6f5 !important; overflow: hidden !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0a1f44, #2e5aad) !important;
    border-radius: 4px !important;
}

/* ════════════════════════════════════════
   SCROLLBAR
════════════════════════════════════════ */
::-webkit-scrollbar { width: 10px !important; height: 10px !important; }
::-webkit-scrollbar-track { background: #d8e2f5 !important; border-radius: 4px !important; }
::-webkit-scrollbar-thumb { background: #2e5aad !important; border-radius: 4px !important; border: 2px solid #d8e2f5 !important; }
::-webkit-scrollbar-thumb:hover { background: #0a1f44 !important; }
::-webkit-scrollbar-corner { background: #d8e2f5 !important; }
* { scrollbar-width: thin !important; scrollbar-color: #2e5aad #d8e2f5 !important; }

/* ════════════════════════════════════════
   MISC
════════════════════════════════════════ */
hr { border-color: #e0e6f5 !important; }
[data-testid="stSpinner"] p { color: #2e5aad !important; font-size: 0.82rem !important; }
[data-testid="stCaptionContainer"], .stCaption, small { color: #4a5e8a !important; }
</style>
""", unsafe_allow_html=True)

CHROMA_DIR = "./bvrith_chroma_db"
SECTION_OPTIONS = [
    "All",
    "About BVRIT",
    "Departments",
    "Admissions",
    "Fee Structure",
    "Placements",
    "Campus & Facilities",
    "Faculty",
    "Contact",
    "Live Site Data",
]

DIM_ORDER = [
    ("Functional",  "01"),
    ("Quality",     "02"),
    ("Safety",      "03"),
    ("Security",    "04"),
    ("Robustness",  "05"),
    ("Performance", "06"),
    ("Context",     "07"),
    ("RAGAS",       "08"),
]

RAGAS_THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.80,
}

# ── Page selector ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:#1a3a6e; border-radius:8px; padding:0.5rem 0.8rem 0.3rem 0.8rem;
                margin-bottom:0.6rem;'>
        <div style='color:#c9a84c; font-size:0.72rem; font-weight:700;
                    letter-spacing:0.08em; text-transform:uppercase;
                    margin-bottom:0.3rem;'>📌 Navigation</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Dashboard", "🔍 Observability"],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Evaluation Dashboard")
    st.caption("Results from `eval_report.json` — run `python run_eval.py` to refresh.")

    REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.json")

    if not os.path.exists(REPORT_FILE):
        st.warning(
            "No evaluation report found. Run `python run_eval.py` first to generate it.",
            icon="⚠️",
        )
        st.code("python run_eval.py", language="bash")
        st.stop()

    with open(REPORT_FILE, encoding="utf-8") as f:
        report = json.load(f)

    summary   = report.get("summary", {})
    dim_data  = report.get("dimension_breakdown", {})
    cases     = report.get("cases", [])
    ragas     = report.get("ragas_scores") or {}
    weakest   = report.get("weakest_dimension", "N/A")
    fix       = report.get("recommended_fix", "")
    generated = report.get("generated_at", "")[:16].replace("T", " ")

    total      = summary.get("total", 0)
    passed     = summary.get("passed", 0)
    warned     = summary.get("warned", 0)
    failed     = summary.get("failed", total - passed - warned)
    pass_rate  = summary.get("pass_rate_pct", 0)
    
    # Backward compatibility: compute weakest dimension if missing
    if weakest == "N/A" and dim_data:
        active_dims = [d for d, _ in DIM_ORDER if d in dim_data]
        if active_dims:
            weakest = min(
                active_dims,
                key=lambda d: dim_data[d].get("passed", 0) / max(dim_data[d].get("total", 1), 1)
            )

    st.caption(f"🕐 Last run: {generated}  |  Judge: `{report.get('judge_model','?')}`  |  Chatbot: `{report.get('chat_model','?')}`")
    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────
    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Cases", total)
    c2.metric("✅ Passed",   passed,  delta=None)
    c3.metric("⚠️ Warning",  warned,  delta=None)
    c4.metric("❌ Failed",   failed,  delta=None)

    rate_color = "normal" if pass_rate >= 80 else ("off" if pass_rate >= 60 else "inverse")
    c5.metric("Pass Rate", f"{pass_rate}%")

    # Pass rate progress bar
    st.markdown(f"**Overall pass rate: {pass_rate}%**")
    st.progress(int(pass_rate) / 100)
    st.divider()

    # ── Per-dimension breakdown ────────────────────────────────────────────
    st.subheader("Per-Dimension Breakdown")

    col_left, col_right = st.columns(2)
    dim_items = [(d, c) for d, c in DIM_ORDER if d in dim_data]

    for idx, (dim, code) in enumerate(dim_items):
        c     = dim_data[dim]
        t     = c.get("total", 0)
        p     = c.get("passed", 0)
        w     = c.get("warned", 0)
        f     = t - p - w
        rate  = round(100 * p / max(t, 1))
        col   = col_left if idx % 2 == 0 else col_right
        flag  = " 🔴" if dim == weakest else ""

        with col:
            st.markdown(f"**{code} {dim}{flag}** — {p}/{t} passed ({rate}%)")
            bar_val = p / max(t, 1)
            st.progress(bar_val)
            verdict_str = f"✅ {p} pass"
            if w: verdict_str += f"  ⚠️ {w} warn"
            if f: verdict_str += f"  ❌ {f} fail"
            st.caption(verdict_str)

    st.divider()

    # ── Weakest dimension & fix ────────────────────────────────────────────
    st.subheader("Weakest Dimension")
    weak_code = next((c for d, c in DIM_ORDER if d == weakest), "??")
    st.error(f"**{weak_code} — {weakest}** has the lowest pass rate", icon="🔴")
    weak_fail = next((r for r in cases if r.get("dimension") == weakest
                      and r.get("verdict","") == "fail"), None)
    if weak_fail:
        st.markdown(f"> Failed on: *\"{weak_fail['question']}\"*")
        st.markdown(f"> {weak_fail['reasoning']}")
    if fix:
        st.info(f"**Recommended fix:** {fix}", icon="🔧")
    st.divider()

    # ── RAGAS scores ───────────────────────────────────────────────────────
    st.subheader("RAGAS Scores")
    if ragas:
        r1, r2, r3, r4 = st.columns(4)
        metrics = [
            (r1, "Faithfulness",      ragas.get("faithfulness")),
            (r2, "Answer Relevancy",  ragas.get("answer_relevancy")),
            (r3, "Context Precision", ragas.get("context_precision")),
            (r4, "Context Recall",    ragas.get("context_recall")),
        ]
        for col, label, val in metrics:
            if isinstance(val, float):
                thresh = RAGAS_THRESHOLDS.get(label.lower().replace(" ", "_"), 0.8)
                delta_val = round(val - thresh, 2)
                col.metric(label, f"{val:.2f}", delta=f"{delta_val:+.2f} vs threshold")
            else:
                col.metric(label, "N/A")

        # Diagnosis
        st.markdown("**RAGAS Diagnosis:**")
        any_issue = False
        diagnoses = {
            "faithfulness":      "Low faithfulness — model generating facts not in context. Tighten grounding rule.",
            "answer_relevancy":  "Low answer relevancy — responses drift off-topic. Check query reformulation.",
            "context_precision": "Context Precision is lowest — retrieval returns irrelevant chunks. Reduce chunk_size or add metadata filters.",
            "context_recall":    "Low context recall — relevant chunks missed. Increase top-k or chunk_overlap.",
        }
        for metric, msg in diagnoses.items():
            val = ragas.get(metric)
            thresh = RAGAS_THRESHOLDS.get(metric, 0.8)
            if isinstance(val, float) and val < thresh:
                st.warning(f"⚠️ **{metric.replace('_',' ').title()} = {val:.2f}** (threshold {thresh}): {msg}")
                any_issue = True
        if not any_issue:
            st.success("All RAGAS metrics are within acceptable range. ✅")
    else:
        st.info(
            "RAGAS scores not computed. Install `ragas` and `datasets`, then re-run `python run_eval.py`.",
            icon="ℹ️",
        )
    st.divider()

    # ── All test cases table ───────────────────────────────────────────────
    st.subheader("All Test Cases")

    # Filter controls
    filter_col1, filter_col2 = st.columns([2, 2])
    dim_options = ["All"] + [d for d, _ in DIM_ORDER if d in dim_data]
    verdict_options = ["All", "✅ pass", "⚠️ warn", "❌ fail"]
    sel_dim     = filter_col1.selectbox("Filter by dimension", dim_options)
    sel_verdict = filter_col2.selectbox("Filter by verdict",   verdict_options)

    filtered = cases
    if sel_dim != "All":
        filtered = [r for r in filtered if r.get("dimension") == sel_dim]
    if sel_verdict != "All":
        v = sel_verdict.split()[-1]
        filtered = [r for r in filtered if r.get("verdict", r.get("passed") and "pass" or "fail") == v]

    rows = []
    for r in filtered:
        verdict = r.get("verdict", "pass" if r.get("passed") else "fail")
        icon    = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(verdict, "❓")
        rows.append({
            "ID":         r["id"],
            "Dimension":  r["dimension"],
            "Question":   r["question"][:80] + ("…" if len(r["question"]) > 80 else ""),
            "Verdict":    f"{icon} {verdict}",
            "Time (s)":   r.get("elapsed_s", 0),
            "Reasoning":  r.get("reasoning", "")[:100] + ("…" if len(r.get("reasoning","")) > 100 else ""),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download buttons
    dl1, dl2 = st.columns(2)
    with dl1:
        with open(REPORT_FILE, encoding="utf-8") as f:
            dl1.download_button("⬇️ Download JSON report", f.read(),
                                file_name="eval_report.json", mime="application/json")
    md_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.md")
    if os.path.exists(md_file):
        with open(md_file, encoding="utf-8") as f:
            dl2.download_button("⬇️ Download Markdown report", f.read(),
                                file_name="eval_report.md", mime="text/markdown")

    st.stop()  # don't render the chat page below


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Observability":
    from observability import (
        get_session_logs, compute_session_stats, check_alerts,
        LATENCY_ALERT_THRESHOLD, COST_ALERT_THRESHOLD,
        ERROR_RATE_ALERT_THRESHOLD, LOG_FILE,
    )

    st.title("🔍 Observability Dashboard")
    st.caption("Live telemetry for every LLM call — latency, cost, token usage, errors, and A/B prompt tracking.")

    # ── Load persistent logs ──────────────────────────────────────────────
    persistent_logs: list[dict] = []
    if LOG_FILE.exists():
        with LOG_FILE.open(encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        persistent_logs.append(json.loads(_line))
                    except json.JSONDecodeError:
                        pass

    session_logs = get_session_logs()

    # Merge: persistent is the full history; session is this run's subset
    all_logs = persistent_logs  # persistent already includes session writes

    if not all_logs:
        st.info(
            "No telemetry recorded yet. Start a conversation on the **💬 Chat** page "
            "and logs will appear here automatically.",
            icon="ℹ️",
        )
        st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader("📈 Summary Metrics")
    stats = compute_session_stats(all_logs)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Calls",   stats["total_queries"])
    m2.metric("Avg Latency",   f"{stats['avg_latency']:.2f}s")
    m3.metric("P95 Latency",   f"{stats['p95_latency']:.2f}s")
    m4.metric("Total Cost",    f"${stats['total_cost']:.4f}")
    m5.metric("Total Tokens",  f"{stats['total_tokens']:,}")
    m6.metric("Errors",        stats["error_count"],
              delta=None if stats["error_count"] == 0 else f"{stats['error_count']} failure(s)")

    # ── Active alerts ──────────────────────────────────────────────────────
    last_entry = all_logs[-1] if all_logs else None
    active_alerts = check_alerts(last_entry=last_entry, logs=all_logs)
    if active_alerts:
        st.divider()
        st.subheader("🚨 Active Alerts")
        for alert in active_alerts:
            st.warning(alert)
    else:
        st.success("✅ No active alerts — all thresholds within range.", icon="✅")

    st.divider()

    # ── Threshold reference ────────────────────────────────────────────────
    with st.expander("⚙️ Alert Thresholds", expanded=False):
        t1, t2, t3 = st.columns(3)
        t1.metric("Latency Threshold",    f"{LATENCY_ALERT_THRESHOLD}s")
        t2.metric("Cost Threshold",       f"${COST_ALERT_THRESHOLD:.2f}")
        t3.metric("Error Rate Threshold", f"{ERROR_RATE_ALERT_THRESHOLD*100:.0f}%")

    # ── Call type breakdown ────────────────────────────────────────────────
    st.subheader("📊 Calls by Type")
    type_counts: dict[str, int] = {}
    type_costs:  dict[str, float] = {}
    for entry in all_logs:
        ct = entry.get("call_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
        type_costs[ct]  = type_costs.get(ct, 0.0) + entry.get("estimated_cost_usd", 0.0)

    if type_counts:
        tc_df = pd.DataFrame([
            {
                "Call Type":  ct,
                "Count":      type_counts[ct],
                "Total Cost": f"${type_costs[ct]:.5f}",
                "Avg Cost":   f"${type_costs[ct]/type_counts[ct]:.5f}",
            }
            for ct in sorted(type_counts, key=lambda k: type_counts[k], reverse=True)
        ])
        st.dataframe(tc_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── A/B Prompt experiment ──────────────────────────────────────────────
    st.subheader("🧪 A/B Prompt Experiment")
    ab_logs = [e for e in all_logs if e.get("prompt_version") in ("A", "B")]
    if ab_logs:
        ab_a = [e for e in ab_logs if e.get("prompt_version") == "A"]
        ab_b = [e for e in ab_logs if e.get("prompt_version") == "B"]

        a_lats = [e["latency_seconds"] for e in ab_a]
        b_lats = [e["latency_seconds"] for e in ab_b]
        a_costs = [e["estimated_cost_usd"] for e in ab_a]
        b_costs = [e["estimated_cost_usd"] for e in ab_b]

        ab1, ab2 = st.columns(2)
        with ab1:
            st.markdown("**Version A (Baseline)**")
            st.metric("Calls",       len(ab_a))
            st.metric("Avg Latency", f"{(sum(a_lats)/len(a_lats)):.2f}s" if a_lats else "—")
            st.metric("Avg Cost",    f"${(sum(a_costs)/len(a_costs)):.5f}" if a_costs else "—")
        with ab2:
            st.markdown("**Version B (Strict grounding)**")
            st.metric("Calls",       len(ab_b))
            st.metric("Avg Latency", f"{(sum(b_lats)/len(b_lats)):.2f}s" if b_lats else "—")
            st.metric("Avg Cost",    f"${(sum(b_costs)/len(b_costs)):.5f}" if b_costs else "—")
    else:
        st.info("No A/B prompt data recorded yet. Calls appear after chatting on the Chat page.", icon="🧪")

    st.divider()

    # ── Recent call log table ──────────────────────────────────────────────
    st.subheader("📋 Recent Call Log")

    # Filter controls
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    all_types   = ["All"] + sorted({e.get("call_type","unknown") for e in all_logs})
    all_status  = ["All", "success", "failure", "rejected_input"]
    sel_type    = fc1.selectbox("Filter by call type", all_types,    key="obs_type")
    sel_status  = fc2.selectbox("Filter by status",    all_status,   key="obs_status")
    n_rows      = fc3.number_input("Rows", min_value=10, max_value=500, value=50, step=10)

    filtered_logs = list(reversed(all_logs))  # most recent first
    if sel_type != "All":
        filtered_logs = [e for e in filtered_logs if e.get("call_type") == sel_type]
    if sel_status != "All":
        filtered_logs = [e for e in filtered_logs if e.get("status") == sel_status]
    filtered_logs = filtered_logs[:int(n_rows)]

    log_rows = []
    for e in filtered_logs:
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        log_rows.append({
            "Timestamp":   ts,
            "Type":        e.get("call_type", "?"),
            "Model":       e.get("model", "?"),
            "Status":      {"success": "✅", "failure": "❌", "rejected_input": "🚫"}.get(
                               e.get("status",""), e.get("status","")),
            "Latency (s)": round(e.get("latency_seconds", 0), 3),
            "In Tokens":   e.get("input_tokens", 0),
            "Out Tokens":  e.get("output_tokens", 0),
            "Cost ($)":    f"{e.get('estimated_cost_usd', 0):.6f}",
            "Prompt Ver":  e.get("prompt_version") or "—",
        })

    if log_rows:
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No entries match the selected filters.")

    # ── Download raw log ───────────────────────────────────────────────────
    st.divider()
    if LOG_FILE.exists():
        with LOG_FILE.open(encoding="utf-8") as _f:
            raw_jsonl = _f.read()
        st.download_button(
            "⬇️ Download full JSONL log",
            data=raw_jsonl,
            file_name="bvrith_chatbot.jsonl",
            mime="application/jsonl",
        )

    st.stop()  # don't render the chat page below


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PAGE (everything below only runs when page == "💬 Chat")
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.6rem 0 0.4rem 0;'>
        <div style='font-size:1.6rem;'>🎓</div>
        <div style='color:#c9a84c; font-size:0.95rem; font-weight:700;
                    letter-spacing:0.05em; text-transform:uppercase;'>BVRIT Hyderabad</div>
        <div style='color:#7a94cc; font-size:0.7rem; margin-top:2px;'>Info Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    index_exists = os.path.isdir(CHROMA_DIR) and any(
        f.endswith(".parquet") or f.endswith(".sqlite3")
        for _, _, files in os.walk(CHROMA_DIR)
        for f in files
    )
    index_status = "🟢 Persisted" if index_exists else "🔴 Not built — run ingest.py"

    # Live chunk count
    chunk_count = "—"
    if index_exists:
        try:
            vs = get_vectorstore()
            chunk_count = vs._collection.count()
        except Exception:
            chunk_count = "error"

    st.metric("Document", "BVRITH_Knowledge_Base.docx")
    st.metric("Chunks indexed", chunk_count)
    st.metric("Index status", index_status)
    st.divider()

    chunk_size = st.slider("Chunk size (tokens)", 200, 1000, 500, step=50)
    overlap = st.slider("Overlap (tokens)", 0, 200, 75, step=25)
    top_k = st.slider("Top-k results", 1, 10, 5)
    section_filter = st.selectbox("Section filter", SECTION_OPTIONS)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reload index"):
        reset_vectorstore()
        st.success("Index cache cleared — will reload on next query.")

    st.divider()

    # ── Memory / Profile panel (Layer 3 + 5) ─────────────────────────────
    st.subheader("🧠 Memory")
    # Ensure user_id exists
    uid = get_or_create_user_id(st.session_state)
    profile = st.session_state.get("profile", {})
    if profile.get("name"):
        st.caption(f"👤 {profile['name']}")
    if profile.get("branch_interest"):
        st.caption(f"🎓 Branch: {profile['branch_interest']}")
    if profile.get("detail_level"):
        st.caption(f"📝 Style: {profile['detail_level']}")

    if st.button("🗑️ Clear my data (Privacy)"):
        msg = handle_clear_data(uid)
        st.session_state.messages = []
        st.session_state.profile = {}
        st.session_state.privacy_shown = False
        st.session_state.pop("user_id", None)
        st.success(msg)
        st.rerun()

    st.divider()
    st.caption("Built with LangChain · ChromaDB · GPT-4o mini · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bvrit-header">
    <div class="bvrit-header-logo">🎓</div>
    <div class="bvrit-header-text">
        <h1>BVRIT Hyderabad — Official Information Assistant</h1>
        <p>College of Engineering for Women &nbsp;|&nbsp; Powered by RAG &amp; GPT-4o mini</p>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Layer 3: Load user profile at session start ───────────────────────────────
uid = get_or_create_user_id(st.session_state)
if "profile" not in st.session_state:
    st.session_state.profile = load_profile(uid)

# ── Layer 5: Show privacy notice on first interaction ─────────────────────────
if "privacy_shown" not in st.session_state:
    st.session_state.privacy_shown = False

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            cols = st.columns(min(len(msg["images"]), 3))
            for i, (caption, url) in enumerate(msg["images"]):
                with cols[i % 3]:
                    st.image(url, caption=caption, use_container_width=True)
        if msg.get("citations"):
            st.caption("📎 Sources: " + " · ".join(msg["citations"]))
        if msg.get("refused"):
            st.warning("⚠️ This answer was outside my knowledge base — I directed you to the college directly.", icon="⚠️")
        if msg.get("elapsed"):
            st.caption(f"⏱ {msg['elapsed']:.2f}s")

# Chat input
if prompt := st.chat_input("Ask about BVRIT Hyderabad…"):

    # ── Layer 5: Show privacy notice on very first user message ───────────
    if not st.session_state.privacy_shown:
        with st.chat_message("assistant"):
            st.markdown(PRIVACY_NOTICE)
        st.session_state.messages.append(
            make_assistant_turn(PRIVACY_NOTICE, [], False)
        )
        st.session_state.privacy_shown = True

    # ── Layer 5: Handle "clear my data" command ───────────────────────────
    if is_clear_data_command(prompt):
        clear_msg = handle_clear_data(uid)
        with st.chat_message("assistant"):
            st.markdown(clear_msg)
        # Reset all session memory
        st.session_state.messages = []
        st.session_state.profile = {}
        st.session_state.privacy_shown = False
        st.session_state.pop("user_id", None)
        st.rerun()

    # ── Layer 1: Add clean user message (NO context block) ────────────────
    st.session_state.messages.append(make_user_turn(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check index exists before querying
    if not index_exists:
        with st.chat_message("assistant"):
            st.error(
                "The knowledge-base index hasn't been built yet. "
                "Please run `python ingest.py` first, then refresh this page."
            )
        st.stop()

    # Generate answer
    with st.chat_message("assistant"):

        # ── Pure image request — skip RAG pipeline entirely ───────────────
        if is_image_only_request(prompt):
            images = detect_image_request(prompt)
            st.markdown(IMAGE_ONLY_RESPONSE)
            if images:
                cols = st.columns(min(len(images), 3))
                for i, (caption, url) in enumerate(images):
                    with cols[i % 3]:
                        st.image(url, caption=caption, use_container_width=True)
                st.caption("📷 Images sourced from bvrithyderabad.edu.in — visit the [gallery](https://bvrithyderabad.edu.in/gallery/) for more.")
            st.session_state.messages.append(
                make_assistant_turn(IMAGE_ONLY_RESPONSE, [], False, 0.0, images)
            )
            st.stop()

        # ── Fast-path: greetings — reply warmly, skip RAG pipeline ──────────
        import random as _random
        _lower_prompt = prompt.lower().strip()
        _profile = st.session_state.profile

        _GREETING_TRIGGERS = {
            # hello variants
            "hi", "hii", "hiii", "hiiii", "hey", "hello", "helo", "hllo", "helo",
            "heya", "heyo", "heyy",
            # good X
            "good morning", "good afternoon", "good evening", "good night",
            "gm", "gn",
            # what's up
            "what's up", "whats up", "wassup", "sup",
            # howdy / namaste
            "howdy", "namaste", "namaskar", "vanakkam",
            # how are you
            "how are you", "how r u", "how are u", "how r you",
            "how's it going", "how is it going", "how do you do",
            "hru",
        }

        def _is_greeting(text: str) -> bool:
            """True when the entire message (ignoring punctuation) is a greeting."""
            clean = text.strip().rstrip("!?.,").lower()
            _QUESTION_WORDS = {"what", "which", "how", "where", "when", "who",
                               "why", "is", "are", "does", "do", "can", "tell",
                               "show", "list", "give", "explain", "fee", "fees",
                               "department", "admission", "placement", "faculty"}
            # exact match
            if clean in _GREETING_TRIGGERS:
                return True
            # starts with a greeting token and is short (≤5 words) with no question words
            words = clean.split()
            if len(words) <= 5:
                for trigger in _GREETING_TRIGGERS:
                    if clean.startswith(trigger):
                        # reject if remaining words contain a question/topic word
                        remaining = words[len(trigger.split()):]
                        if not any(w in _QUESTION_WORDS for w in remaining):
                            return True
            return False

        if _is_greeting(_lower_prompt):
            _name = _profile.get("name")
            _branch = _profile.get("branch_interest")

            # Pick a time-appropriate opener for good-morning/evening etc.
            _lower_clean = _lower_prompt.rstrip("!?., ")
            if "morning" in _lower_clean or _lower_clean == "gm":
                _opener = "Good morning"
            elif "afternoon" in _lower_clean:
                _opener = "Good afternoon"
            elif "evening" in _lower_clean:
                _opener = "Good evening"
            elif "night" in _lower_clean or _lower_clean == "gn":
                _opener = "Good night"
            else:
                _opener = _random.choice(["Hello", "Hi there", "Hey", "Hi"])

            # Personalise with name if known
            _greeting = f"{_opener}, **{_name}**! 👋" if _name else f"{_opener}! 👋"

            # Tailor the follow-up based on what we know
            if _branch:
                _follow = (
                    f"I remember you're interested in **{_branch}** at BVRIT Hyderabad. "
                    f"Feel free to ask me about admissions, fees, placements, faculty, or anything else!"
                )
            else:
                _follow = (
                    "I'm the BVRIT Hyderabad Information Assistant. "
                    "Ask me anything about admissions, departments, fees, placements, "
                    "facilities, or faculty!"
                )

            _greet_reply = f"{_greeting}\n\n{_follow}"
            st.markdown(_greet_reply)
            st.session_state.messages.append(
                make_assistant_turn(_greet_reply, [], False, 0.0)
            )
            st.rerun()

        # ── Detect "I am telling you my name/branch" (declaration) ──────────
        import re as _re
        _name_declared = _re.search(
            r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z\s]{0,30}?)(?:\s*[,.]|$)",
            _lower_prompt,
        )
        _branch_declared = _re.search(
            r"(?:my branch is|i(?:'m| am) (?:interested in|studying|from)|interested in)\s+"
            r"(cse[-\s]?aiml|cse|ece|eee|it\b|information technology|computer science|"
            r"electronics|electrical|aiml|ai\s*&?\s*ml)",
            _lower_prompt,
        )

        _declaration_reply_parts = []

        if _name_declared:
            _extracted_name = _name_declared.group(1).strip().title()
            # Filter out false positives like "i am interested in cse"
            _stop_words = {"interested", "from", "studying", "a", "an", "the", "student",
                           "here", "looking", "planning", "going", "want"}
            if _extracted_name.lower().split()[0] not in _stop_words:
                _profile["name"] = _extracted_name
                _profile["user_id"] = uid
                st.session_state.profile = _profile
                save_profile(_profile)
                _declaration_reply_parts.append(f"Got it! I'll remember your name is **{_extracted_name}**.")

        if _branch_declared:
            _raw_branch = _branch_declared.group(1).strip().upper()
            # Normalise common variants
            _branch_map = {
                "CSE-AIML": "CSE-AIML", "CSE AIML": "CSE-AIML", "AIML": "CSE-AIML",
                "AI&ML": "CSE-AIML", "AI ML": "CSE-AIML",
                "COMPUTER SCIENCE": "CSE", "CSE": "CSE",
                "ELECTRONICS": "ECE", "ECE": "ECE",
                "ELECTRICAL": "EEE", "EEE": "EEE",
                "INFORMATION TECHNOLOGY": "IT", "IT": "IT",
            }
            _norm_branch = _branch_map.get(_raw_branch, _raw_branch)
            _profile["branch_interest"] = _norm_branch
            _profile["user_id"] = uid
            st.session_state.profile = _profile
            save_profile(_profile)
            _declaration_reply_parts.append(
                f"Noted! I'll remember you're interested in **{_norm_branch}**."
            )

        if _declaration_reply_parts:
            _decl_reply = " ".join(_declaration_reply_parts) + (
                " Feel free to ask me anything about BVRIT Hyderabad!"
            )
            st.markdown(_decl_reply)
            st.session_state.messages.append(make_assistant_turn(_decl_reply, [], False, 0.0))
            st.rerun()

        # ── Detect "asking what their saved name/branch is" (recall question) ──
        _asking_name = any(p in _lower_prompt for p in [
            "what is my name", "what's my name", "do you know my name",
            "remember my name", "who am i", "tell me my name", "my name?",
        ])
        _asking_branch = any(p in _lower_prompt for p in [
            "what is my branch", "what's my branch", "what branch am i",
            "which branch am i", "remember my branch", "my branch?",
        ])
        if _asking_name or _asking_branch:
            parts = []
            if _asking_name:
                if _profile.get("name"):
                    parts.append(f"Your name is **{_profile['name']}**.")
                else:
                    parts.append("I don't have your name saved yet — just tell me and I'll remember it!")
            if _asking_branch:
                if _profile.get("branch_interest"):
                    parts.append(f"Your branch of interest is **{_profile['branch_interest']}**.")
                else:
                    parts.append("I don't have your branch preference saved yet — just tell me!")
            _recall_reply = " ".join(parts)
            st.markdown(_recall_reply)
            st.session_state.messages.append(make_assistant_turn(_recall_reply, [], False, 0.0))
            st.rerun()

        # ── Normal RAG query ───────────────────────────────────────────────
        with st.spinner("Retrieving and generating answer…"):
            start = time.time()
            try:
                # ── Layer 1+2: Build clean history for LLM (with summary if applicable) ──
                llm_history = get_llm_history_with_summary(st.session_state.messages[:-1])

                # ── Layer 4: Build personalization prefix and inject into question ──
                persona_prefix = build_personalization_prefix(st.session_state.profile)

                # We inject personalization into the question so run_rag_pipeline
                # sees it in its system prompt via chat_history's first assistant slot.
                # Alternatively: modify rag_pipeline to accept a system_prefix kwarg.
                # Here we prepend to the first system message via the history trick:
                # Add a "virtual" system message as first assistant turn when profile exists.
                if persona_prefix:
                    persona_turn = {
                        "role": "assistant",
                        "content": f"[PERSONALIZATION CONTEXT]\n{persona_prefix}",
                    }
                    llm_history = [persona_turn] + llm_history

                response, citations, refused = run_rag_pipeline(
                    question=prompt,
                    chat_history=llm_history,
                    top_k=top_k,
                    section_filter=section_filter,
                )
                elapsed = time.time() - start
            except Exception as e:
                response = f"⚠️ An error occurred: {e}"
                citations, refused, elapsed = [], False, 0.0

        # Check if the RAG answer also warrants showing images
        images = detect_image_request(prompt)

        st.markdown(response)
        if images:
            st.markdown("**📸 Here are some photos from BVRIT Hyderabad:**")
            cols = st.columns(min(len(images), 3))
            for i, (caption, url) in enumerate(images):
                with cols[i % 3]:
                    st.image(url, caption=caption, use_container_width=True)
            st.caption("📷 Images sourced from bvrithyderabad.edu.in — visit the [gallery](https://bvrithyderabad.edu.in/gallery/) for more.")
        if citations:
            st.caption("📎 Sources: " + " · ".join(citations))
        if refused:
            st.warning("⚠️ This answer was outside my knowledge base — I directed you to the college directly.", icon="⚠️")
        st.caption(f"⏱ {elapsed:.2f}s")

    # ── Layer 1: Persist clean assistant turn ────────────────────────────
    st.session_state.messages.append(
        make_assistant_turn(response, citations, refused, elapsed, images)
    )

    # ── Layer 2: Maybe compress old turns into a summary ─────────────────
    st.session_state.messages = maybe_summarize(st.session_state.messages)

    # ── Layer 3+4: Update user profile whenever the user mentions personal info ──
    # Check every turn (not just every 5th) so name/branch are never lost if
    # the user closes the tab before reaching turn 5.
    PERSONAL_TRIGGERS = [
        "my name is", "i am ", "i'm ", "call me", "my branch", "i prefer",
        "i want", "i like", "my interest", "interested in", "studying",
        "i study", "tell me in", "brief", "detailed",
    ]
    should_update = any(t in prompt.lower() for t in PERSONAL_TRIGGERS)
    user_turn_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    # Also update every 5 turns as a catch-all
    if should_update or (user_turn_count > 0 and user_turn_count % 5 == 0):
        st.session_state.profile = extract_and_update_profile(
            st.session_state.messages, st.session_state.profile
        )
        st.session_state.profile["user_id"] = uid
        save_profile(st.session_state.profile)
