"""
app.py
------
Phase 4: Streamlit Chat UI for BVRIT Hyderabad RAG Chatbot.

Run:  streamlit run app.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Ensure working directory is the script's own folder ──────────────────────
# This fixes issues when Streamlit is launched from a different directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import run_rag_pipeline, get_vectorstore, reset_vectorstore

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
    page_title="BVRITH Info Assistant",
    page_icon="🎓",
    layout="wide",
)

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
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
    st.caption("Built with LangChain · ChromaDB · GPT-4o mini · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎓 BVRIT Hyderabad — College Information Assistant")
st.caption(
    "Ask me anything about BVRIT Hyderabad College of Engineering for Women — "
    "admissions, fees, departments, placements, facilities, faculty, and more."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
if prompt := st.chat_input("Ask about BVRITH…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": IMAGE_ONLY_RESPONSE,
                "citations": [],
                "refused": False,
                "elapsed": 0.0,
                "images": images,
            })
            st.stop()

        # ── Normal RAG query ───────────────────────────────────────────────
        with st.spinner("Retrieving and generating answer…"):
            start = time.time()
            try:
                response, citations, refused = run_rag_pipeline(
                    question=prompt,
                    chat_history=st.session_state.messages[:-1],
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

    # Persist in session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "citations": citations,
            "refused": refused,
            "elapsed": elapsed,
            "images": images,
        }
    )
