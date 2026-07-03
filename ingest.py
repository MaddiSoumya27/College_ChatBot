"""
ingest.py
---------
Phase 1: Load KB markdown files → chunk → embed → persist to Chroma.
Reads markdown files directly — no dependency on BVRITH_Knowledge_Base.docx.

Run once:  python ingest.py
Re-check:  python ingest.py --check
"""

import argparse
import os
import re
import shutil
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bvrith_chroma_db")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
LAST_VERIFIED = str(date.today())

# ── KB markdown files in order ────────────────────────────────────────────────
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")
KB_FILES = [
    ("About BVRIT",        "00_college_overview.md"),
    ("Departments",        "02_departments.md"),
    ("Admissions",         "01_admissions.md"),
    ("Fee Structure",      "01_admissions.md"),   # fee subsection extracted below
    ("Placements",         "03_placements.md"),
    ("Campus & Facilities","04_facilities_research_governance.md"),
    ("Faculty",            "05_faculty_directory.md"),
    ("Contact",            "04_facilities_research_governance.md"),  # contact subsection
    ("Live Site Data",     "07_corpus_live_data.md"),
]

# Subsections to extract for split sections
EXTRACT_MAP = {
    "Fee Structure": "Fees",
    "Contact": "Contact Details",
}


def infer_section(text: str) -> str:
    """
    Guess which H1 section a chunk belongs to based on keyword matches.
    Falls back to 'General' if none match.
    """
    text_lower = text.lower()
    scores = {}
    keyword_map = {
        "About BVRIT": ["naac", "aicte", "vision", "mission", "established", "about bvrit", "jntuh", "autonomous",
                        "sri vishnu educational society", "women empowerment", "chairman", "accreditation"],
        "Departments": ["department", "cse", "ece", "eee", "it", "bs&h", "aiml", "b.tech", "m.tech",
                        "intake", "undergraduate", "postgraduate", "computer science", "electronics"],
        "Admissions": ["admission", "eamcet", "ecet", "seat", "category a", "category b",
                       "tseamcet", "tsche", "eapcet", "rank", "hostel", "transport"],
        "Fee Structure": ["fee", "tuition", "₹", "lakh", "per year", "batch", "rupees"],
        "Placements": ["placement", "lpa", "recruiter", "tap cell", "package", "median salary", "placed",
                       "multinational", "mnc", "campus recruitment"],
        "Campus & Facilities": ["library", "gym", "cafeteria", "facility", "research", "innovation", "patent",
                                "infrastructure", "lab", "laboratory", "sports", "hostel capacity",
                                "hostel security", "lady warden", "anti-ragging", "security officer",
                                "500+ students", "4 blocks", "150+ rooms"],
        "Faculty": ["faculty", "professor", "hod", "principal", "ph.d", "experience", "publications",
                    "dr.", "mr.", "ms.", "associate professor", "assistant professor",
                    "k.v.n. sunitha", "sunitha", "founder principal", "b.tech (ece", "m.tech (cs",
                    "guided", "ph.d scholars", "textbooks", "dst-tide", "dst-nims",
                    "29 years", "30+ years", "patents filed", "research areas"],
        "Contact": ["contact", "phone", "email", "address", "fax", "hyderabad – 500",
                    "+91 40", "info@bvrit", "bvrithyderabad.edu.in"],
        "Live Site Data": ["source: https://bvrithyderabad", "live site data", "crawled directly",
                           "post author", "post published", "post category", "flash new"],
    }
    for section, keywords in keyword_map.items():
        scores[section] = sum(1 for kw in keywords if kw in text_lower)

    # Tie-breaking: if Faculty and Departments tie, prefer Faculty when bio-specific
    # keywords are present
    bio_markers = ["founder principal", "b.tech (ece", "m.tech (cs", "ph.d scholars",
                   "guided", "k.v.n. sunitha", "sunitha", "dst-tide"]
    if scores.get("Faculty", 0) > 0 and any(m in text_lower for m in bio_markers):
        scores["Faculty"] += 3  # boost Faculty section for principal/faculty bios

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def infer_subsection(text: str) -> str:
    """
    Try to pick up an H2-style heading from the chunk text.
    Returns the first heading-like line found, or empty string.
    """
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 80 and line[0].isupper() and not line.endswith("."):
            # rough heuristic: short capitalised line without sentence-ending period
            return line
    return ""


def extract_subsection(md_text: str, heading: str) -> str:
    """Return content under a specific ## heading until the next ## heading."""
    lines = md_text.splitlines()
    collecting = False
    result = []
    pattern = re.compile(r"^##\s+(.+)$")
    for line in lines:
        m = pattern.match(line)
        if m:
            if heading.lower() in m.group(1).lower():
                collecting = True
                result.append(line)
                continue
            elif collecting:
                break
        if collecting:
            result.append(line)
    return "\n".join(result)


def load_kb_documents() -> list[Document]:
    """
    Load all KB markdown files and return as LangChain Documents,
    one per section, with section metadata pre-assigned.
    """
    docs = []
    seen_files: dict[str, str] = {}  # filepath → full text cache

    for section, filename in KB_FILES:
        filepath = os.path.join(KB_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠️  Missing KB file: {filepath} — skipping section '{section}'")
            continue

        if filepath not in seen_files:
            with open(filepath, encoding="utf-8") as f:
                seen_files[filepath] = f.read()
        md_text = seen_files[filepath]

        # Extract subsection if needed
        extract_heading = EXTRACT_MAP.get(section)
        if extract_heading:
            content = extract_subsection(md_text, extract_heading)
        else:
            content = md_text

        if not content.strip():
            print(f"  ⚠️  Empty content for section '{section}' — skipping")
            continue

        docs.append(Document(
            page_content=content,
            metadata={"section_hint": section, "source": filename}
        ))

    print(f"  Loaded {len(docs)} section documents from {KB_DIR}")
    return docs


def build_index():
    print(f"Loading KB markdown files from {KB_DIR} …")
    docs = load_kb_documents()
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"  Total chars: {total_chars}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    # Attach metadata per chunk
    section_counters: dict[str, int] = {}
    for chunk in chunks:
        # Use section_hint from the document metadata as primary signal
        hint = chunk.metadata.get("section_hint", "")
        section = hint if hint else infer_section(chunk.page_content)
        subsection = infer_subsection(chunk.page_content)
        section_counters[section] = section_counters.get(section, 0) + 1
        idx = section_counters[section]
        slug = section.lower().replace(" & ", "_").replace(" ", "_")
        chunk.metadata.update(
            {
                "source": chunk.metadata.get("source", "kb"),
                "section": section,
                "subsection": subsection,
                "chunk_id": f"{slug}_{idx:03d}",
                "last_verified": LAST_VERIFIED,
            }
        )

    print("\nChunk distribution by section:")
    for sec, count in sorted(section_counters.items()):
        print(f"  {sec:<25} {count:>3} chunks")

    print("\nEmbedding and persisting to Chroma …")
    embed_kwargs = {"model": EMBED_MODEL}
    if OPENAI_BASE_URL:
        embed_kwargs["openai_api_base"] = OPENAI_BASE_URL
    embeddings = OpenAIEmbeddings(**embed_kwargs)

    # Always wipe the old index before rebuilding to avoid duplicate chunks.
    # If the directory is locked (app running), use a timestamped temp dir
    # and update the symlink/path — user must restart app after ingest.
    if os.path.isdir(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f"  Wiped existing index at {CHROMA_DIR}")
        except PermissionError:
            # DB is locked by a running process — write to a fresh directory
            # and rename after. User must restart the app.
            import tempfile, time
            alt_dir = CHROMA_DIR + "_new"
            if os.path.isdir(alt_dir):
                shutil.rmtree(alt_dir, ignore_errors=True)
            CHROMA_DIR_USE = alt_dir
            print(f"  ⚠️  Old index locked — writing new index to {alt_dir}")
            print(f"  ⚠️  Stop the app, then rename '{alt_dir}' → 'bvrith_chroma_db'")
        else:
            CHROMA_DIR_USE = CHROMA_DIR
    else:
        CHROMA_DIR_USE = CHROMA_DIR

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR_USE
    )
    count = vectorstore._collection.count()
    print(f"\n✅  Indexed {count} chunks → {CHROMA_DIR_USE}")
    if CHROMA_DIR_USE != CHROMA_DIR:
        print(f"\n⚠️  ACTION REQUIRED:")
        print(f"   1. Stop the Streamlit app (Ctrl+C)")
        print(f"   2. In PowerShell run:")
        print(f"      Remove-Item -Recurse -Force bvrith_chroma_db")
        print(f"      Rename-Item bvrith_chroma_db_new bvrith_chroma_db")
        print(f"   3. Run: streamlit run app.py")
    return count


def check_index():
    embed_kwargs = {"model": EMBED_MODEL}
    if OPENAI_BASE_URL:
        embed_kwargs["openai_api_base"] = OPENAI_BASE_URL
    embeddings = OpenAIEmbeddings(**embed_kwargs)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    count = vectorstore._collection.count()
    print(f"Index contains {count} chunks in {CHROMA_DIR}")

    # Quick smoke test
    test_queries = [
        "What is the fee for CSE?",
        "Who is the principal?",
        "What is the average placement package?",
    ]
    print("\nSmoke-test retrieval:")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    for q in test_queries:
        results = retriever.invoke(q)
        sections_hit = [r.metadata.get("section", "?") for r in results]
        print(f"  Q: {q}")
        print(f"     Sections retrieved: {sections_hit}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check existing index only")
    args = parser.parse_args()

    if args.check:
        check_index()
    else:
        build_index()
