"""
run_eval.py
-----------
Phase 5: Run all test cases against the live chatbot and score with an LLM judge.
Also runs RAGAS on the G1/G2/G3 cases.

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
from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline import run_rag_pipeline, get_vectorstore

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o")   # intentionally different from chatbot's gpt-4o-mini
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
TEST_CASES_FILE = "test_cases.json"
REPORT_JSON = "eval_report.json"
REPORT_MD = "eval_report.md"


# ── LLM Judge ────────────────────────────────────────────────────────────────
def judge_response(
    question: str,
    actual_answer: str,
    expected_answer: str,
    pass_criteria: str,
) -> tuple[bool, str]:
    """
    Returns (passed: bool, reasoning: str).
    Uses a separate LLM (JUDGE_MODEL) to avoid self-evaluation bias.
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
  "passed": true or false
  "reasoning": one sentence explaining your decision

JSON only, no extra text."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        result = json.loads(raw)
        return bool(result.get("passed")), str(result.get("reasoning", ""))
    except Exception:
        # Fallback: simple keyword check
        passed = "pass" in raw.lower() or "true" in raw.lower()
        return passed, raw[:200]


# ── RAGAS evaluation ──────────────────────────────────────────────────────────
def run_ragas_eval(ragas_cases: list[dict]) -> Optional[dict]:
    """
    Run RAGAS on the G1/G2/G3 cases.
    Returns a dict with metric scores, or None if RAGAS unavailable.
    """
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

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)


# ── Main runner ───────────────────────────────────────────────────────────────
def run_all():
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        all_cases = json.load(f)

    print(f"Loaded {len(all_cases)} test cases from {TEST_CASES_FILE}")
    print(f"Using judge model: {JUDGE_MODEL}\n")

    results = []
    dimension_counts: dict[str, dict] = {}

    # Multi-turn session state tracking
    sessions: dict[str, list[dict]] = {}

    for case in all_cases:
        dim = case.get("dimension", "Unknown")
        cid = case.get("id", "?")
        question = case.get("question", "")
        pass_criteria = case.get("pass_criteria", "")
        expected = case.get("expected_answer", "")

        # Handle multi-turn context
        session_id = case.get("session_id")
        turn = case.get("turn", 1)
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            chat_history = sessions[session_id]
        else:
            chat_history = []

        print(f"  [{cid}] {dim}: {question[:60]}…" if len(question) > 60 else f"  [{cid}] {dim}: '{question}'")

        start = time.time()
        try:
            actual_answer, citations, refused = run_rag_pipeline(
                question=question,
                chat_history=chat_history,
            )
            elapsed = time.time() - start
        except Exception as e:
            actual_answer = f"ERROR: {e}"
            citations, refused, elapsed = [], False, 0.0

        # Update multi-turn session
        if session_id:
            sessions[session_id].append({"role": "user", "content": question})
            sessions[session_id].append({"role": "assistant", "content": actual_answer})

        # Performance check
        perf_pass = elapsed < 10.0

        # Judge
        passed, reasoning = judge_response(question, actual_answer, expected, pass_criteria)

        # For performance cases, also gate on timing
        if dim == "Performance":
            passed = passed and perf_pass

        record = {
            "id": cid,
            "dimension": dim,
            "question": question,
            "actual_answer": actual_answer,
            "expected_answer": expected,
            "pass_criteria": pass_criteria,
            "citations": citations,
            "refused": refused,
            "elapsed_s": round(elapsed, 2),
            "passed": passed,
            "reasoning": reasoning,
        }
        results.append(record)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"     {status} ({elapsed:.1f}s) — {reasoning[:80]}")

        if dim not in dimension_counts:
            dimension_counts[dim] = {"passed": 0, "total": 0}
        dimension_counts[dim]["total"] += 1
        if passed:
            dimension_counts[dim]["passed"] += 1

    # RAGAS
    ragas_cases = [c for c in all_cases if c.get("dimension") == "RAGAS"]
    print("\nRunning RAGAS scoring …")
    ragas_scores = run_ragas_eval(ragas_cases)

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = round(100 * total_passed / total, 1) if total else 0

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": total_passed,
            "failed": total - total_passed,
            "pass_rate_pct": pass_rate,
        },
        "dimension_breakdown": dimension_counts,
        "ragas_scores": ragas_scores,
        "cases": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅  Saved {REPORT_JSON}")

    # ── Generate Markdown report ──────────────────────────────────────────
    weakest = min(
        dimension_counts,
        key=lambda d: dimension_counts[d]["passed"] / max(dimension_counts[d]["total"], 1),
    ) if dimension_counts else "N/A"

    md_lines = [
        "## BVRITH Chatbot — Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Summary:** Total test cases: {total} | Passed: {total_passed} | "
        f"Failed: {total - total_passed} | Pass rate: {pass_rate}%",
        "",
        "**Per-dimension breakdown**",
        "| Dimension | Passed | Total |",
        "|---|---|---|",
    ]
    for dim, counts in sorted(dimension_counts.items()):
        md_lines.append(f"| {dim} | {counts['passed']} | {counts['total']} |")

    md_lines += [
        "",
        f"**Weakest dimension:** {weakest}",
    ]

    # Recommend fix for weakest
    fixes = {
        "Functional": "Verify chunk metadata — ensure section labels are correct so retrieval pulls the right chunks.",
        "Quality": "Check source doc figures against official site and update BVRITH_Knowledge_Base.docx.",
        "Safety": "Tighten the system prompt RULES section to explicitly prohibit outcome guarantees.",
        "Security": "Add a pre-filter to detect prompt-injection patterns before sending to the LLM.",
        "Robustness": "Add input validation in run_rag_pipeline() to handle empty/gibberish inputs gracefully.",
        "Performance": "Increase ChromaDB index size or reduce top-k; consider caching embeddings.",
        "Context": "Extend chat_history window and ensure history is passed correctly to run_rag_pipeline.",
        "RAGAS": "Improve chunk_overlap or increase top-k to ensure full context is retrieved.",
    }
    fix = fixes.get(weakest, "Review failing cases and adjust prompt / chunking strategy.")
    md_lines.append(f"**Recommended fix:** {fix}")

    if ragas_scores:
        md_lines += [
            "",
            "**RAGAS scores:**",
            f"- Faithfulness: {ragas_scores.get('faithfulness', 'N/A')}",
            f"- Answer Relevancy: {ragas_scores.get('answer_relevancy', 'N/A')}",
            f"- Context Precision: {ragas_scores.get('context_precision', 'N/A')}",
            f"- Context Recall: {ragas_scores.get('context_recall', 'N/A')}",
        ]
    else:
        md_lines.append("\n**RAGAS scores:** Not computed (install `ragas` and `datasets` to enable).")

    md_lines += [
        "",
        "---",
        "### Failed Cases",
        "",
    ]
    failed = [r for r in results if not r["passed"]]
    if failed:
        for r in failed:
            md_lines += [
                f"**[{r['id']}] {r['dimension']}**",
                f"- Question: {r['question']}",
                f"- Criteria: {r['pass_criteria']}",
                f"- Reasoning: {r['reasoning']}",
                "",
            ]
    else:
        md_lines.append("🎉 All cases passed!")

    md_content = "\n".join(md_lines)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅  Saved {REPORT_MD}")
    print(f"\n{'='*50}")
    print(f"  PASS RATE: {pass_rate}%  ({total_passed}/{total})")
    print(f"  WEAKEST:   {weakest}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_all()
