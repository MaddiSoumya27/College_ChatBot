"""
build_corpus_md.py
------------------
Converts bvrith_corpus/*.json (live-crawled pages) into a structured
markdown file at kb/07_corpus_live_data.md, which build_kb_docx.py then
includes as the "9. Live Site Data" section.

Run:  python build_corpus_md.py
"""

import json
import os
import glob
import re

CORPUS_DIR = "./bvrith_corpus"
OUTPUT_MD = "./kb/07_corpus_live_data.md"

# Map filename prefix → subsection heading
TOPIC_MAP = [
    ("management",                     "Management & Leadership"),
    ("under-graduate",                 "Undergraduate Programs"),
    ("post-graduate",                  "Postgraduate Programs"),
    ("admission-process",              "Admission Process"),
    ("contact-dr",                     "Admissions Contact"),
    ("home",                           "College Overview (Live)"),
    ("synergia",                       "Events — Synergia 2026"),
    ("iot-and-smart",                  "Events — IoT & MATLAB Workshop"),
    ("distinguished-lecture",          "Events — Distinguished Lecture"),
    ("annual-community",               "Events — Annual Community Conference"),
    ("one-week-national",              "Events — National FDP"),
    ("best-csi",                       "Awards — Best CSI Supporting Principal"),
    ("pm-vidyalaxmi",                  "Schemes — PM Vidyalaxmi"),
    ("institutions-innovation",        "Events — IIC Sustainability Seminar"),
]


def slug_to_heading(filename: str) -> str:
    """Fallback: convert filename slug to title-case heading."""
    name = filename.replace(".json", "").replace("-", " ").replace("_", " ")
    return name.title()


def clean_text(text: str) -> str:
    """Remove boilerplate navigation lines and tidy whitespace."""
    skip_patterns = [
        r"^Skip to content$",
        r"^Know more about$",
        r"^VISIT THE DEPARTMENT$",
        r"^view more$",
        r"^More news$",
        r"^More awards$",
        r"^Learn about placements$",
        r"^\d+$",              # bare numbers from JS counters
        r"^\+$",
        r"^0$",
        r"^life at bvrith$",
        r"^Campus Tour$",
        r"^our visionaries$",
        r"^Alumni speak$",
        r"^know about us$",
        r"^Placement partners$",
        r"^What.s happening at BVRITH$",
        r"^Awards & recognitions$",
        r"^Building.*Generation.*Women.*Leaders",
        r"^Empowering Women to$",
        r"^Engineer a Better Future",
        r"^Engineering a World of$",
        r"^quality and Opportunity$",
        r"^for Women\.$",
        r"^Years of$",
        r"^academic excellence$",
        r"^Programmes$",
        r"^Faculties$",
        r"^Students placed$",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in skip_patterns):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def get_heading(filename: str) -> str:
    basename = os.path.basename(filename).replace(".json", "")
    for prefix, heading in TOPIC_MAP:
        if prefix in basename:
            return heading
    return slug_to_heading(basename)


def build_md():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.json")))
    if not files:
        print(f"⚠️  No JSON files found in {CORPUS_DIR}")
        return

    lines = [
        "# Live Site Data",
        "",
        "Content crawled directly from bvrithyderabad.edu.in.",
        "",
    ]

    for filepath in files:
        heading = get_heading(filepath)
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)

        url = data.get("url", "")
        title = data.get("title", heading)
        text = clean_text(data.get("text", ""))

        # Skip if text after cleaning is too sparse (< 5 lines of real content)
        cleaned_lines = [l for l in text.splitlines() if len(l.strip()) > 10]
        if len(cleaned_lines) < 4:
            print(f"  ⚠️  Skipping {heading} — too little content after cleaning ({len(cleaned_lines)} lines)")
            continue

        lines.append(f"## {heading}")
        lines.append("")
        if url:
            lines.append(f"Source: {url}")
            lines.append("")

        # Strip leading title line if it duplicates the heading
        text_lines = text.splitlines()
        if text_lines and title.lower() in text_lines[0].lower():
            text_lines = text_lines[1:]
        text = "\n".join(text_lines).strip()

        lines.append(text)
        lines.append("")

    md_content = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    print(f"✅  Written: {OUTPUT_MD}")
    print(f"   Sections: {md_content.count(chr(10) + '## ')}")
    print(f"   Total chars: {len(md_content)}")


if __name__ == "__main__":
    build_md()
