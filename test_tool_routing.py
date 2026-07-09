"""
test_tool_routing.py
--------------------
Tests the three-capability routing in run_rag_pipeline():
  1. RAG (ChromaDB document retrieval + GPT grounded generation)
  2. fee_calculator (structured fee breakdown)
  3. date_checker (deadline / event date comparison against today)

For each of six queries this script records:
  - Which capability the model chose
  - Arguments extracted (for tool-based capabilities)
  - The final answer

Run:
    python test_tool_routing.py
"""

import os
import sys
import textwrap
from datetime import date

# ── Monkey-patch _try_fee_calculator and _try_date_checker to record routing ──
import rag_pipeline as _rp
from fee_calculator import FEE_CALCULATOR_TOOL
from date_checker import DATE_CHECKER_TOOL, check_date, format_date_result, KNOWN_EVENTS

_routing_log: dict = {}   # filled per query during the test run
_current_query_id: list = [None]   # mutable cell so inner functions can write to it


# ── Wrap _try_fee_calculator ──────────────────────────────────────────────────
_orig_fee = _rp._try_fee_calculator

def _patched_fee(question, llm):
    result = _orig_fee(question, llm)
    qid = _current_query_id[0]
    if result is not None and qid is not None:
        _routing_log[qid] = {
            "capability": "fee_calculator",
            "tool_name":  "calculate_fee",
            "args":       _extract_fee_args(question),
        }
    return result

def _extract_fee_args(question: str) -> dict:
    """Re-run the Python parser to show what args were extracted."""
    program = _rp._extract_program_from_question(question)
    flags   = _rp._parse_fee_flags(question)
    return {"program": program, **flags}

_rp._try_fee_calculator = _patched_fee


# ── Wrap _try_date_checker ────────────────────────────────────────────────────
_orig_date = _rp._try_date_checker

def _patched_date(question, llm):
    result = _orig_date(question, llm)
    qid = _current_query_id[0]
    if result is not None and qid is not None:
        event_key = _rp._extract_event_key(question)
        _routing_log[qid] = {
            "capability": "date_checker",
            "tool_name":  "check_date",
            "args": {
                "event_date": event_key or "<resolved from question>",
                "event_name": KNOWN_EVENTS.get(event_key, (None, None))[0]
                               if event_key else None,
            },
        }
    return result

_rp._try_date_checker = _patched_date


# ── Also wrap run_rag_pipeline to catch RAG fallback ─────────────────────────
_orig_rag = _rp.run_rag_pipeline

def _patched_rag(question, chat_history=None, **kwargs):
    chat_history = chat_history or []
    result = _orig_rag(question, chat_history, **kwargs)
    qid = _current_query_id[0]
    # If nothing else has recorded routing info, it went through RAG
    if qid is not None and qid not in _routing_log:
        _routing_log[qid] = {
            "capability": "RAG",
            "tool_name":  "chromadb_retriever + gpt_grounded_generation",
            "args":       {"top_k": kwargs.get("top_k", _rp.TOP_K)},
        }
    return result


# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "id": "Q1",
        "label": "Fee calculation — CSE with hostel",
        "query": "What is the total fee for CSE including hostel for year 1?",
    },
    {
        "id": "Q2",
        "label": "Date check — EAPCET counselling",
        "query": "Has the EAPCET counselling deadline passed?",
    },
    {
        "id": "Q3",
        "label": "RAG — general placement info",
        "query": "What is the highest placement package at BVRIT Hyderabad?",
    },
    {
        "id": "Q4",
        "label": "Date check — days until Synergia fest",
        "query": "How many days are left until Synergia 2026?",
    },
    {
        "id": "Q5",
        "label": "Fee calculation — ECE 4-year total",
        "query": "What is the total fee for the entire 4-year ECE program?",
    },
    {
        "id": "Q6",
        "label": "RAG — departments offered",
        "query": "Which departments are available at BVRIT Hyderabad?",
    },
]


def divider(char="─", width=70):
    return char * width


def run_tests():
    print(divider("═"))
    print("  BVRIT Hyderabad Chatbot — Tool Routing Test")
    print(f"  Today: {date.today().strftime('%d %B %Y')}")
    print(divider("═"))

    results = []

    for test in TEST_QUERIES:
        qid   = test["id"]
        label = test["label"]
        query = test["query"]

        print(f"\n{divider()}")
        print(f"  {qid}: {label}")
        print(f"  Query : \"{query}\"")
        print(divider())

        _current_query_id[0] = qid
        _routing_log.pop(qid, None)  # clear any stale entry

        try:
            answer, citations, refused = _patched_rag(query, chat_history=[])
        except Exception as exc:
            print(f"  ❌ ERROR: {exc}")
            results.append({
                "id": qid, "label": label, "query": query,
                "capability": "ERROR", "args": {}, "answer": str(exc),
            })
            continue

        routing = _routing_log.get(qid, {
            "capability": "RAG",
            "tool_name":  "chromadb_retriever + gpt_grounded_generation",
            "args":       {"top_k": _rp.TOP_K},
        })

        print(f"  Capability : {routing['capability']}")
        print(f"  Tool used  : {routing['tool_name']}")
        print(f"  Args       : {routing['args']}")
        print()
        print("  Answer:")
        # Indent and wrap the answer for readability
        wrapped = textwrap.fill(answer, width=66,
                                initial_indent="    ",
                                subsequent_indent="    ")
        # If the answer is multi-line markdown, don't hard-wrap it
        if "\n" in answer:
            for line in answer.splitlines():
                print(f"    {line}")
        else:
            print(wrapped)

        if citations:
            print(f"\n  Citations: {citations}")
        if refused:
            print("  ⚠️  Response was a refusal.")

        results.append({
            "id":         qid,
            "label":      label,
            "query":      query,
            "capability": routing["capability"],
            "tool":       routing["tool_name"],
            "args":       routing["args"],
            "answer":     answer[:300] + ("…" if len(answer) > 300 else ""),
            "refused":    refused,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{divider('═')}")
    print("  SUMMARY")
    print(divider("═"))
    print(f"  {'ID':<4} {'Capability':<22} {'Label'}")
    print(f"  {divider('-',4)} {divider('-',22)} {divider('-',38)}")
    for r in results:
        icon = {"RAG": "📄", "fee_calculator": "💰", "date_checker": "📅",
                "ERROR": "❌"}.get(r["capability"], "?")
        print(f"  {r['id']:<4} {icon} {r['capability']:<20} {r['label']}")
    print(divider("═"))
    print()

    return results


if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Export it or add to .env")
        sys.exit(1)

    run_tests()
