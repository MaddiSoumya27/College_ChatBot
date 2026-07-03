"""
app.py
------
Phase 4: Streamlit Chat UI for BVRIT Hyderabad RAG Chatbot.

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
page = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "📊 Dashboard"],
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
# CHAT PAGE (everything below only runs when page == "💬 Chat")
# ══════════════════════════════════════════════════════════════════════════════

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
