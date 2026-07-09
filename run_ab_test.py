"""
run_ab_test.py
--------------
Exercise 4 — Controlled A/B comparison of the two grounding prompt variants.

Runs a fixed set of 10 test questions through BOTH versions A and B
(20 total calls), then prints a comparison table and appends results to
OBSERVABILITY_NOTES.md.

Usage:
    python run_ab_test.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import run_rag_pipeline
from observability import get_session_logs, get_prompt_suffix

# ── 10 fixed test questions (mix of clearly-answerable + borderline) ──────────
AB_TEST_QUESTIONS: list[dict] = [
    # Clearly answerable from KB
    {"id": 1, "question": "What is the annual tuition fee for the CSE program?",
     "answerable": True},
    {"id": 2, "question": "Who is the principal of BVRIT Hyderabad and what are her qualifications?",
     "answerable": True},
    {"id": 3, "question": "What is the highest placement package recorded at BVRIT Hyderabad?",
     "answerable": True},
    {"id": 4, "question": "What M.Tech programs does BVRIT Hyderabad offer?",
     "answerable": True},
    {"id": 5, "question": "How many students can the hostel accommodate?",
     "answerable": True},
    {"id": 6, "question": "What departments are available at BVRIT Hyderabad?",
     "answerable": True},
    {"id": 7, "question": "What is the contact phone number for BVRIT Hyderabad?",
     "answerable": True},
    # Borderline / not clearly in doc
    {"id": 8, "question": "What is the pass percentage for CSE students in the 2024 semester exams?",
     "answerable": False},
    {"id": 9, "question": "Does BVRIT Hyderabad offer an exchange program with international universities?",
     "answerable": False},
    {"id": 10, "question": "What is the average CGPA of placed students from the ECE department?",
     "answerable": False},
]

REFUSAL_PHRASES = [
    "i don't have that information",
    "i don't have that specific information",
    "not in my knowledge base",
    "please contact",
    "cannot answer",
    "outside my knowledge",
    "i don't have the complete information",
]

CITATION_PATTERN = re.compile(r"\[[^\]]+,\s*Chunk\s*\d+\]", re.IGNORECASE)


def has_citation(text: str) -> bool:
    return bool(CITATION_PATTERN.search(text))


def is_refused(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


def run_controlled_ab() -> list[dict]:
    """Run all 10 questions through both versions; return 20 result dicts."""
    results = []
    for ver in ["A", "B"]:
        print(f"\n{'='*60}")
        print(f"  Running Version {ver} ({len(AB_TEST_QUESTIONS)} questions)")
        print(f"{'='*60}")
        for item in AB_TEST_QUESTIONS:
            qid = item["id"]
            question = item["question"]
            answerable = item["answerable"]

            print(f"  [{ver}-{qid}] {question[:70]}...")
            t0 = time.perf_counter()
            try:
                answer, citations, refused_flag = run_rag_pipeline(
                    question=question,
                    chat_history=[],
                    prompt_version=ver,
                )
                latency = time.perf_counter() - t0
                cited = has_citation(answer)
                refused_detected = is_refused(answer) or refused_flag
            except Exception as exc:
                answer = f"ERROR: {exc}"
                latency = time.perf_counter() - t0
                cited = False
                refused_detected = False

            result = {
                "version":      ver,
                "question_id":  qid,
                "question":     question,
                "answerable":   answerable,
                "answer":       answer,
                "has_citation": cited,
                "refused":      refused_detected,
                "latency_s":    round(latency, 2),
            }

            # For version B refusals on answerable questions, classify as CORRECT/INCORRECT
            if ver == "B" and refused_detected:
                result["refusal_classification"] = (
                    "INCORRECT — doc has the answer but Version B was too strict"
                    if answerable else
                    "CORRECT — question is genuinely not answerable from the KB"
                )
            else:
                result["refusal_classification"] = "N/A"

            results.append(result)
            print(f"       cited={cited}  refused={refused_detected}  latency={latency:.2f}s")

    return results


def build_comparison_table(results: list[dict]) -> str:
    """Build a markdown comparison table from the 20 results."""
    # Map: question_id → {A: ..., B: ...}
    by_q: dict[int, dict] = {}
    for r in results:
        qid = r["question_id"]
        if qid not in by_q:
            by_q[qid] = {}
        by_q[qid][r["version"]] = r

    header = (
        "| Q# | Question | Version | Has Citation | Refused | Refusal Classification |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for qid in sorted(by_q.keys()):
        q_text = by_q[qid].get("A", by_q[qid].get("B", {})).get("question", "")
        q_short = q_text[:70] + ("…" if len(q_text) > 70 else "")
        for ver in ["A", "B"]:
            r = by_q[qid].get(ver, {})
            cited   = "✅" if r.get("has_citation") else "❌"
            refused = "✅" if r.get("refused") else "❌"
            refusal_class = r.get("refusal_classification", "N/A")
            rows.append(f"| {qid} | {q_short} | {ver} | {cited} | {refused} | {refusal_class} |")

    return header + "\n".join(rows)


def summarize_ab(results: list[dict]) -> str:
    a_results = [r for r in results if r["version"] == "A"]
    b_results = [r for r in results if r["version"] == "B"]

    a_cited   = sum(1 for r in a_results if r["has_citation"])
    b_cited   = sum(1 for r in b_results if r["has_citation"])
    a_refused = sum(1 for r in a_results if r["refused"])
    b_refused = sum(1 for r in b_results if r["refused"])

    n = len(a_results)

    # Classify B refusals
    b_correct_refusals   = [r for r in b_results if r["refused"] and not r["answerable"]]
    b_incorrect_refusals = [r for r in b_results if r["refused"] and r["answerable"]]

    lines = [
        f"**Summary:**",
        f"",
        f"- **Citations:** Version A produced citations on {a_cited}/{n} questions; "
        f"Version B on {b_cited}/{n}.",
        f"  Version {'B' if b_cited > a_cited else 'A'} produced more citations.",
        f"",
        f"- **Refusals:** Version A refused {a_refused}/{n} questions; "
        f"Version B refused {b_refused}/{n}.",
        f"  Version {'B' if b_refused > a_refused else 'A'} refused more often.",
        f"",
        f"- **Version B refusal classifications:**",
    ]

    if b_correct_refusals:
        lines.append(f"  - CORRECT refusals ({len(b_correct_refusals)}):")
        for r in b_correct_refusals:
            lines.append(f"    - Q{r['question_id']}: \"{r['question'][:70]}\"")
    if b_incorrect_refusals:
        lines.append(f"  - INCORRECT refusals ({len(b_incorrect_refusals)}) — KB has the answer but B was too strict:")
        for r in b_incorrect_refusals:
            lines.append(f"    - Q{r['question_id']}: \"{r['question'][:70]}\"")
    if not b_refused:
        lines.append("  - No refusals recorded for Version B.")

    return "\n".join(lines)


def append_to_notes(table: str, summary: str, results: list[dict]) -> None:
    notes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OBSERVABILITY_NOTES.md")
    section = f"""
---

## Exercise 4 — A/B Test on Grounding Prompt

**Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Implementation

- **Version A**: The original Day 4 grounding prompt (unchanged baseline).
- **Version B**: Version A + strict citation and refusal rules:
  _"Cite [Section, Page] for every fact. If the exact answer is not in the context, say 'I don't have that specific information.' Never infer or extrapolate."_
- Each production query is randomly assigned A or B (50/50 via `random.choice`).
- The assigned version is logged in the `prompt_version` field of every `rag_generation` log entry.
- Controlled comparison: 10 fixed questions run through BOTH versions (20 total calls).

### 10 Test Questions

Mix: 7 clearly answerable from the KB, 3 borderline/not in doc.

### Comparison Table

{table}

### Analysis

{summary}

---
"""
    mode = "a" if os.path.exists(notes_path) else "w"
    with open(notes_path, mode, encoding="utf-8") as f:
        f.write(section)
    print(f"\nResults appended to OBSERVABILITY_NOTES.md")


if __name__ == "__main__":
    print("BVRIT Hyderabad Chatbot — Exercise 4 A/B Test")
    print(f"Running {len(AB_TEST_QUESTIONS)} questions × 2 versions = {len(AB_TEST_QUESTIONS)*2} total calls\n")

    results = run_controlled_ab()

    print("\n" + "="*60)
    print("  COMPARISON TABLE")
    print("="*60)
    table = build_comparison_table(results)
    print(table)

    summary = summarize_ab(results)
    print("\n" + summary)

    # Save raw results to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to ab_test_results.json")

    append_to_notes(table, summary, results)
    print("\nDone.")
