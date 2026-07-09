"""
run_eval.py
-----------
Phase 5: Run all test cases against the live chatbot, score with an LLM judge,
and produce a structured evaluation report.

Run:  python run_eval.py
Output: eval_report.json  +  eval_report.md
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from rag_pipeline import run_rag_pipeline, get_vectorstore

JUDGE_MODEL     = os.getenv("JUDGE_MODEL", "openai/gpt-4o")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
TEST_CASES_FILE = "test_cases.json"
REPORT_JSON     = "eval_report.json"
REPORT_MD       = "eval_report.md"

# Canonical dimension order with codes
DIM_ORDER = [
    ("Functional",   "01"),
    ("Quality",      "02"),
    ("Safety",       "03"),
    ("Security",     "04"),
    ("Robustness",   "05"),
    ("Performance",  "06"),
    ("Context",      "07"),
    ("RAGAS",        "08"),
]

# Per-dimension recommended fixes
FIXES = {
    "Functional":  "Verify chunk metadata — ensure section labels are correct so retrieval pulls the right chunks.",
    "Quality":     "Cross-check source figures against the official site and update the KB markdown files.",
    "Safety":      "Tighten system prompt RULES — explicitly prohibit outcome guarantees and negative comparisons.",
    "Security":    "Strengthen system prompt with explicit injection-defence instructions and add input sanitisation before the LLM call.",
    "Robustness":  "Add input validation in run_rag_pipeline() to gracefully handle empty, gibberish, and non-English inputs.",
    "Performance": "Reduce top-k for simple queries; cache embeddings; consider async retrieval for multi-section questions.",
    "Context":     "Extend chat_history window and add coreference resolution so follow-up questions resolve correctly.",
    "RAGAS":       "Increase chunk_overlap or top-k to ensure complete context is retrieved; add metadata filters to reduce noise.",
}

# RAGAS thresholds for diagnosis
RAGAS_THRESHOLDS = {
    "faithfulness":      (0.85, "Low faithfulness — model is generating facts not present in retrieved chunks. Tighten the grounding rule in the system prompt."),
    "answer_relevancy":  (0.80, "Low answer relevancy — responses drift off-topic. Check that query reformulation isn't injecting unrelated keywords."),
    "context_precision": (0.75, "Context Precision is lowest — retrieval returns some irrelevant chunks. Consider reducing chunk_size or adding metadata filters."),
    "context_recall":    (0.80, "Low context recall — relevant chunks are being missed. Increase top-k or chunk_overlap, or improve the section classifier."),
}


# ── LLM Judge ────────────────────────────────────────────────────────────────
def judge_response(
    question: str,
    actual_answer: str,
    expected_answer: str,
    pass_criteria: str,
) -> tuple[str, str]:
    """
    Returns (verdict: 'pass'|'warn'|'fail', reasoning: str).
    'warn' = partially correct / borderline.
    """
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        base_url=OPENAI_BASE_URL or None,
    )
    prompt = f"""You are an impartial evaluator for a college information chatbot.

Question asked: {question}

Pass criteria: {pass_criteria}

Expected answer (reference): {expected_answer}

Actual chatbot response:
\"\"\"
{actual_answer}
\"\"\"

Evaluate whether the actual response passes the criteria.
Reply with a JSON object with exactly two fields:
  "verdict": one of "pass", "warn", or "fail"
    - "pass" = fully meets the criteria
    - "warn" = partially correct, minor omission, or borderline
    - "fail" = clearly does not meet the criteria
  "reasoning": one concise sentence explaining your decision

JSON only, no extra text."""

    from observability import logged_llm_call
    response = logged_llm_call(llm.invoke, "judge", input=[HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        result = json.loads(raw)
        verdict = str(result.get("verdict", "fail")).lower().strip()
        if verdict not in ("pass", "warn", "fail"):
            verdict = "pass" if "pass" in verdict or "true" in verdict else "fail"
        return verdict, str(result.get("reasoning", ""))
    except Exception:
        verdict = "pass" if ("pass" in raw.lower() or "true" in raw.lower()) else "fail"
        return verdict, raw[:200]


# ── RAGAS evaluation ──────────────────────────────────────────────────────────
def run_ragas_eval(ragas_cases: list[dict]) -> Optional[dict]:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("⚠️  ragas / datasets not installed — skipping RAGAS scoring.")
        return None

    questions, answers, contexts, ground_truths = [], [], [], []
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    for case in ragas_cases:
        q = case["question"]
        answer, _, _ = run_rag_pipeline(q, [])
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(case.get("expected_answer", ""))

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)


# ── Markdown report builder ───────────────────────────────────────────────────
def build_md_report(
    results: list[dict],
    dimension_counts: dict,
    ragas_scores: Optional[dict],
    generated_at: str,
) -> str:

    total       = len(results)
    total_pass  = sum(1 for r in results if r["verdict"] == "pass")
    total_warn  = sum(1 for r in results if r["verdict"] == "warn")
    total_fail  = sum(1 for r in results if r["verdict"] == "fail")
    pass_rate   = round(100 * total_pass / total, 1) if total else 0

    active_dims = [d for d, _ in DIM_ORDER if d in dimension_counts]

    def pass_ratio(dim):
        c = dimension_counts.get(dim, {})
        return c.get("passed", 0) / max(c.get("total", 1), 1)

    weakest = min(active_dims, key=pass_ratio) if active_dims else "N/A"
    weakest_code = next((code for d, code in DIM_ORDER if d == weakest), "??")
    weak_fail = next((r for r in results if r["dimension"] == weakest and r["verdict"] == "fail"), None)

    lines = []

    # ── Title ──────────────────────────────────────────────────────────────
    lines += [
        "# BVRIT Hyderabad — Chatbot Evaluation Report",
        "",
        f"**Generated:** {generated_at}  |  "
        f"**Judge:** `{JUDGE_MODEL}`  |  "
        f"**Chatbot:** `{CHAT_MODEL}`",
        "",
        "---",
        "",
    ]

    # ── Summary ────────────────────────────────────────────────────────────
    lines += [
        "## Summary",
        "",
        f"| Total test cases | ✅ Passed | ⚠️ Warning | ❌ Failed | Pass rate |",
        f"|:---:|:---:|:---:|:---:|:---:|",
        f"| **{total}** | **{total_pass}** | **{total_warn}** | **{total_fail}** | **{pass_rate}%** |",
        "",
    ]

    # ── Per-dimension breakdown ────────────────────────────────────────────
    lines += [
        "## Per-Dimension Breakdown",
        "",
        "| Code | Dimension | ✅ Pass | ⚠️ Warn | ❌ Fail | Total | Pass Rate |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for dim, code in DIM_ORDER:
        if dim not in dimension_counts:
            continue
        c    = dimension_counts[dim]
        t    = c.get("total", 0)
        p    = c.get("passed", 0)
        w    = c.get("warned", 0)
        f    = t - p - w
        rate = f"{round(100 * p / max(t, 1))}%"
        flag = " 🔴" if dim == weakest else ""
        lines.append(f"| {code} | {dim}{flag} | {p} | {w} | {f} | {t} | {rate} |")
    lines.append("")

    # ── Weakest dimension & fix ────────────────────────────────────────────
    fix = FIXES.get(weakest, "Review failing cases and adjust prompt / chunking strategy.")
    lines += [
        "## Weakest Dimension",
        "",
        f"**{weakest_code} — {weakest}** 🔴",
        "",
    ]
    if weak_fail:
        lines += [
            f"> The chatbot failed on: *\"{weak_fail['question']}\"*",
            f"> {weak_fail['reasoning']}",
            "",
        ]
    lines += [
        f"**Recommended fix:** {fix}",
        "",
        "---",
        "",
    ]

    # ── RAGAS scores ───────────────────────────────────────────────────────
    lines += ["## RAGAS Scores", ""]

    if ragas_scores:
        fa = ragas_scores.get("faithfulness",      "N/A")
        ar = ragas_scores.get("answer_relevancy",  "N/A")
        cp = ragas_scores.get("context_precision", "N/A")
        cr = ragas_scores.get("context_recall",    "N/A")

        def fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else str(v)

        def status(metric, val):
            if not isinstance(val, float):
                return "—"
            return "✅" if val >= RAGAS_THRESHOLDS[metric][0] else "⚠️"

        lines += [
            f"| Metric | Score | Status |",
            f"|:---|:---:|:---:|",
            f"| Faithfulness      | {fmt(fa)} | {status('faithfulness', fa)} |",
            f"| Answer Relevancy  | {fmt(ar)} | {status('answer_relevancy', ar)} |",
            f"| Context Precision | {fmt(cp)} | {status('context_precision', cp)} |",
            f"| Context Recall    | {fmt(cr)} | {status('context_recall', cr)} |",
            "",
        ]

        # Diagnosis — flag metrics below threshold
        diagnoses = []
        for metric, (threshold, msg) in RAGAS_THRESHOLDS.items():
            val = ragas_scores.get(metric)
            if isinstance(val, float) and val < threshold:
                diagnoses.append(f"- ⚠️ **{metric.replace('_', ' ').title()} = {val:.2f}** (threshold {threshold}): {msg}")

        lines += ["**RAGAS Diagnosis:**", ""]
        if diagnoses:
            lines += diagnoses
        else:
            lines.append("All RAGAS metrics are within acceptable range. ✅")
        lines.append("")
    else:
        lines += [
            "> RAGAS scores not computed.",
            "> Install dependencies: `pip install ragas datasets`  then re-run `python run_eval.py`",
            "",
        ]

    lines += ["---", ""]

    # ── All test cases table ───────────────────────────────────────────────
    lines += [
        "## All Test Cases",
        "",
        "| ID | Dimension | Question | Verdict | Time |",
        "|:---:|:---|:---|:---:|:---:|",
    ]
    for r in results:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(r["verdict"], "❓")
        q    = r["question"][:60].replace("|", "\\|") + ("…" if len(r["question"]) > 60 else "")
        lines.append(f"| {r['id']} | {r['dimension']} | {q} | {icon} | {r['elapsed_s']}s |")
    lines.append("")

    # ── Failed / warned detail ─────────────────────────────────────────────
    failed = [r for r in results if r["verdict"] == "fail"]
    warned = [r for r in results if r["verdict"] == "warn"]

    if failed or warned:
        lines += ["---", "", "## Cases Requiring Attention", ""]

    if failed:
        lines += ["### ❌ Failed Cases", ""]
        for r in failed:
            ans_preview = r["actual_answer"][:300] + ("…" if len(r["actual_answer"]) > 300 else "")
            lines += [
                f"#### [{r['id']}] {r['dimension']}",
                f"- **Question:** {r['question']}",
                f"- **Pass criteria:** {r['pass_criteria']}",
                f"- **Actual answer:** {ans_preview}",
                f"- **Judge reasoning:** {r['reasoning']}",
                f"- **Response time:** {r['elapsed_s']}s",
                "",
            ]

    if warned:
        lines += ["### ⚠️ Warning Cases", ""]
        for r in warned:
            lines += [
                f"#### [{r['id']}] {r['dimension']}",
                f"- **Question:** {r['question']}",
                f"- **Pass criteria:** {r['pass_criteria']}",
                f"- **Judge reasoning:** {r['reasoning']}",
                f"- **Response time:** {r['elapsed_s']}s",
                "",
            ]

    if not failed and not warned:
        lines += ["---", "", "## 🎉 All Cases Passed!", ""]

    return "\n".join(lines)


# ── Main runner ───────────────────────────────────────────────────────────────
def run_all():
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        all_cases = json.load(f)

    print(f"{'='*60}")
    print(f"  BVRIT Chatbot Evaluation")
    print(f"  Test cases : {len(all_cases)}  ({TEST_CASES_FILE})")
    print(f"  Judge      : {JUDGE_MODEL}")
    print(f"  Chatbot    : {CHAT_MODEL}")
    print(f"{'='*60}\n")

    results: list[dict] = []
    dimension_counts: dict[str, dict] = {}
    sessions: dict[str, list[dict]] = {}

    for case in all_cases:
        dim      = case.get("dimension", "Unknown")
        cid      = case.get("id", "?")
        question = case.get("question", "")
        criteria = case.get("pass_criteria", "")
        expected = case.get("expected_answer", "")

        session_id = case.get("session_id")
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            chat_history = sessions[session_id]
        else:
            chat_history = []

        q_display = (question[:55] + "…") if len(question) > 55 else f"'{question}'"
        print(f"  [{cid:>10}] {dim:<13} {q_display}")

        start = time.time()
        try:
            actual_answer, citations, refused = run_rag_pipeline(
                question=question,
                chat_history=chat_history,
            )
            elapsed = round(time.time() - start, 2)
        except Exception as e:
            actual_answer = f"ERROR: {e}"
            citations, refused, elapsed = [], False, 0.0

        if session_id:
            sessions[session_id].append({"role": "user",      "content": question})
            sessions[session_id].append({"role": "assistant", "content": actual_answer})

        # Performance gate: must respond within 10s
        perf_ok = elapsed < 10.0

        verdict, reasoning = judge_response(question, actual_answer, expected, criteria)

        if dim == "Performance" and not perf_ok:
            verdict   = "fail"
            reasoning = f"Exceeded 10s limit ({elapsed}s). " + reasoning

        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(verdict, "❓")
        print(f"             {icon} {verdict.upper():<5} ({elapsed}s) — {reasoning[:65]}")

        record = {
            "id":              cid,
            "dimension":       dim,
            "question":        question,
            "actual_answer":   actual_answer,
            "expected_answer": expected,
            "pass_criteria":   criteria,
            "citations":       citations,
            "refused":         refused,
            "elapsed_s":       elapsed,
            "verdict":         verdict,
            "passed":          verdict == "pass",
            "reasoning":       reasoning,
        }
        results.append(record)

        if dim not in dimension_counts:
            dimension_counts[dim] = {"passed": 0, "warned": 0, "total": 0}
        dimension_counts[dim]["total"] += 1
        if verdict == "pass":
            dimension_counts[dim]["passed"] += 1
        elif verdict == "warn":
            dimension_counts[dim]["warned"] += 1

    # ── RAGAS ──────────────────────────────────────────────────────────────
    ragas_cases = [c for c in all_cases if c.get("dimension") == "RAGAS"]
    print(f"\n{'-'*60}")
    print("Running RAGAS scoring …")
    ragas_scores = run_ragas_eval(ragas_cases)

    # ── Totals ──────────────────────────────────────────────────────────────
    total      = len(results)
    total_pass = sum(1 for r in results if r["verdict"] == "pass")
    total_warn = sum(1 for r in results if r["verdict"] == "warn")
    total_fail = sum(1 for r in results if r["verdict"] == "fail")
    pass_rate  = round(100 * total_pass / total, 1) if total else 0

    active_dims = [d for d, _ in DIM_ORDER if d in dimension_counts]
    weakest = min(
        active_dims,
        key=lambda d: dimension_counts[d]["passed"] / max(dimension_counts[d]["total"], 1),
    ) if active_dims else "N/A"

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Save JSON ────────────────────────────────────────────────────────────
    report_data = {
        "generated_at":       datetime.now().isoformat(),
        "judge_model":        JUDGE_MODEL,
        "chat_model":         CHAT_MODEL,
        "summary": {
            "total":          total,
            "passed":         total_pass,
            "warned":         total_warn,
            "failed":         total_fail,
            "pass_rate_pct":  pass_rate,
        },
        "dimension_breakdown":  dimension_counts,
        "weakest_dimension":    weakest,
        "recommended_fix":      FIXES.get(weakest, ""),
        "ragas_scores":         ragas_scores,
        "cases":                results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"✅  Saved {REPORT_JSON}")

    # ── Save Markdown ────────────────────────────────────────────────────────
    md = build_md_report(results, dimension_counts, ragas_scores, generated_at)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅  Saved {REPORT_MD}")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PASS RATE   {pass_rate}%   "
          f"({total_pass} pass / {total_warn} warn / {total_fail} fail  out of {total})")
    print(f"  WEAKEST     {weakest}")
    print(f"  FIX         {FIXES.get(weakest,'')[:55]}…")
    print(f"\n  Per-dimension:")
    for dim, code in DIM_ORDER:
        if dim not in dimension_counts:
            continue
        c   = dimension_counts[dim]
        t   = c.get("total", 0)
        p   = c.get("passed", 0)
        w   = c.get("warned", 0)
        f   = t - p - w
        bar = "█" * p + "▒" * w + "░" * f
        mk  = " ◀ weakest" if dim == weakest else ""
        print(f"  {code} {dim:<14} [{bar}] {p}/{t}{mk}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()
