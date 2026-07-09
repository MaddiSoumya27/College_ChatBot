"""
observability.py
----------------
Production observability layer for the BVRIT Hyderabad RAG Chatbot.

Implements:
  Exercise 1 — logged_llm_call()     : one structured log entry per LLM call
  Exercise 2 — session_stats()       : sidebar metric helpers
  Exercise 3 — alert thresholds + input validator
  Exercise 4 — A/B grounding prompt assignment + prompt variants
  (Exercise 5 is purely analytical — written into OBSERVABILITY_NOTES.md)

Design rule: ALL LLM calls anywhere in the app MUST be routed through
logged_llm_call(). No raw .invoke() calls should survive outside this module.
"""

from __future__ import annotations

import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ── Persistent log file ────────────────────────────────────────────────────────
LOGS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "bvrith_chatbot.jsonl"

# ── In-session log list (shared mutable singleton) ────────────────────────────
# app.py reads this list to build the sidebar dashboard (Exercise 2).
# Access via get_session_logs() — do NOT import _SESSION_LOGS directly.
_SESSION_LOGS: list[dict] = []


def get_session_logs() -> list[dict]:
    """Return the live in-session log list (read-only reference)."""
    return _SESSION_LOGS


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1 — Pricing constants (real OpenAI rates, per 1 000 tokens)
# ══════════════════════════════════════════════════════════════════════════════
# Source: https://openai.com/pricing  (retrieved July 2025)
#   gpt-4o-mini  : $0.150 / 1M input  = $0.00015 / 1K input
#                  $0.600 / 1M output = $0.00060 / 1K output
#   gpt-4o       : $2.500 / 1M input  = $0.00250 / 1K input
#                  $10.00 / 1M output = $0.01000 / 1K output
#   text-embedding-3-small: $0.020 / 1M = $0.00002 / 1K (both directions)

PRICE_TABLE: dict[str, dict[str, float]] = {
    # ---------- OpenAI native keys ----------
    "gpt-4o-mini":                  {"input": 0.00015,  "output": 0.00060},
    "gpt-4o":                       {"input": 0.00250,  "output": 0.01000},
    "gpt-4o-2024-08-06":            {"input": 0.00250,  "output": 0.01000},
    "gpt-4-turbo":                  {"input": 0.01000,  "output": 0.03000},
    "gpt-3.5-turbo":                {"input": 0.00050,  "output": 0.00150},
    "text-embedding-3-small":       {"input": 0.00002,  "output": 0.00002},
    "text-embedding-3-large":       {"input": 0.00013,  "output": 0.00013},
    # ---------- OpenRouter prefix keys ------
    "openai/gpt-4o-mini":           {"input": 0.00015,  "output": 0.00060},
    "openai/gpt-4o":                {"input": 0.00250,  "output": 0.01000},
    "openai/gpt-3.5-turbo":        {"input": 0.00050,  "output": 0.00150},
}

_DEFAULT_PRICE = {"input": 0.00015, "output": 0.00060}  # fall back to gpt-4o-mini rates


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost from real per-1K-token rates.
    cost = (input_tokens/1000 * input_rate) + (output_tokens/1000 * output_rate)
    """
    rates = PRICE_TABLE.get(model, _DEFAULT_PRICE)
    return (input_tokens / 1000.0 * rates["input"]) + (output_tokens / 1000.0 * rates["output"])


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1 — Core logging wrapper
# ══════════════════════════════════════════════════════════════════════════════

def logged_llm_call(
    fn: Callable,
    call_type: str,
    *,
    prompt_version: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Wrap any LLM call and record one structured log entry.

    Parameters
    ----------
    fn          : The callable to invoke (e.g. llm.invoke).
    call_type   : Tag for this call — one of:
                  "rag_generation", "tool_call", "summarization",
                  "query_reformulation", "coreference_resolution",
                  "profile_extraction", "judge"
    prompt_version : "A" or "B" for Exercise 4 A/B test (optional).
    **kwargs    : Passed directly to fn().

    Returns
    -------
    The result of fn(**kwargs), or re-raises the exception after logging.

    Log entry fields (written to both _SESSION_LOGS and LOG_FILE):
        timestamp, call_type, model, input_tokens, output_tokens,
        latency_seconds, estimated_cost_usd, status, prompt_version,
        error_message (on failure only)
    """
    t0 = time.perf_counter()
    status = "success"
    error_msg: Optional[str] = None
    result = None
    input_tokens = 0
    output_tokens = 0
    model = "unknown"

    try:
        result = fn(**kwargs)
        latency = time.perf_counter() - t0

        # ── Extract token usage from the response ──────────────────────────
        usage = None
        if hasattr(result, "usage_metadata"):
            usage = result.usage_metadata
            if isinstance(usage, dict):
                input_tokens  = usage.get("input_tokens",  0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
            elif usage is not None:
                # Object-style (older LangChain versions)
                input_tokens  = getattr(usage, "input_tokens",  0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0

        # Fall back to response_metadata if usage_metadata gave zeros
        if input_tokens == 0 and hasattr(result, "response_metadata"):
            rm = result.response_metadata or {}
            tu = rm.get("token_usage") or rm.get("usage") or {}
            if isinstance(tu, dict):
                input_tokens  = tu.get("prompt_tokens",     0) or 0
                output_tokens = tu.get("completion_tokens", 0) or 0

        # ── Extract model name ─────────────────────────────────────────────
        if hasattr(result, "response_metadata"):
            rm = result.response_metadata or {}
            model = rm.get("model_name") or rm.get("model") or "unknown"
        if model == "unknown" and hasattr(fn, "__self__"):
            model = getattr(fn.__self__, "model_name", "unknown")

    except Exception as exc:
        latency = time.perf_counter() - t0
        status = "failure"
        error_msg = f"{type(exc).__name__}: {exc}"
        raise  # re-raise so callers can handle it

    finally:
        cost = _compute_cost(model, input_tokens, output_tokens)

        entry: dict[str, Any] = {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "call_type":          call_type,
            "model":              model,
            "input_tokens":       input_tokens,
            "output_tokens":      output_tokens,
            "latency_seconds":    round(latency, 4),
            "estimated_cost_usd": round(cost, 8),
            "status":             status,
            "prompt_version":     prompt_version,
        }
        if error_msg:
            entry["error_message"] = error_msg

        # Append to in-session list
        _SESSION_LOGS.append(entry)

        # Append to persistent JSONL file (one JSON object per line)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # never let logging break the app

    return result


def log_rejected_input(user_message: str) -> None:
    """
    Log a rejected user input (too long / invalid) without calling the LLM.
    status = "rejected_input"; no tokens consumed, no cost.
    """
    entry: dict[str, Any] = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "call_type":          "input_validation",
        "model":              "none",
        "input_tokens":       0,
        "output_tokens":      0,
        "latency_seconds":    0.0,
        "estimated_cost_usd": 0.0,
        "status":             "rejected_input",
        "prompt_version":     None,
        "rejected_length":    len(user_message),
    }
    _SESSION_LOGS.append(entry)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2 — Session stats helpers for the sidebar dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _real_calls(logs: list[dict]) -> list[dict]:
    """Filter out rejected-input entries — they have no latency/cost to average."""
    return [e for e in logs if e.get("status") not in ("rejected_input",)]


def compute_session_stats(logs: Optional[list[dict]] = None) -> dict:
    """
    Compute dashboard metrics from the in-session log list.

    Returns a dict with:
        total_queries, avg_latency, p95_latency, total_cost,
        total_tokens, error_count
    All values are float/int — callers format them for display.
    """
    if logs is None:
        logs = _SESSION_LOGS

    real = _real_calls(logs)

    total_queries = len(real)
    latencies = [e["latency_seconds"] for e in real]
    costs     = [e["estimated_cost_usd"] for e in real]
    tokens    = [e["input_tokens"] + e["output_tokens"] for e in real]
    errors    = sum(1 for e in real if e.get("status") == "failure")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_cost  = sum(costs)
    total_tokens_val = sum(tokens)

    # P95 latency — computed from the actual sorted distribution
    p95_latency = 0.0
    if latencies:
        sorted_lat = sorted(latencies)
        idx = max(0, int(0.95 * len(sorted_lat)) - 1)
        p95_latency = sorted_lat[idx]

    return {
        "total_queries":  total_queries,
        "avg_latency":    round(avg_latency, 3),
        "p95_latency":    round(p95_latency, 3),
        "total_cost":     round(total_cost, 6),
        "total_tokens":   total_tokens_val,
        "error_count":    errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 3 — Alert thresholds + input validator
# ══════════════════════════════════════════════════════════════════════════════

# Named constants — single source of truth; referenced in app.py display strings
LATENCY_ALERT_THRESHOLD    = 10.0   # seconds per single call
COST_ALERT_THRESHOLD       = 0.10   # USD per single query
ERROR_RATE_ALERT_THRESHOLD = 0.05   # fraction over rolling last-20-query window
MAX_INPUT_CHARS            = 2000   # hard cap on user message length

# Rolling window size for error-rate calculation
ERROR_RATE_WINDOW = 20


def validate_input(user_message: str) -> tuple[bool, str]:
    """
    Validate user input before it reaches the LLM.

    Returns
    -------
    (is_valid, feedback_message)
    If is_valid is False, the caller should show feedback_message to the user
    and call log_rejected_input() — the LLM must NOT be invoked.
    """
    if len(user_message) > MAX_INPUT_CHARS:
        feedback = (
            f"⚠️ Your message is **{len(user_message):,} characters** — "
            f"the limit is **{MAX_INPUT_CHARS:,} characters**. "
            "Please shorten your question and try again."
        )
        return False, feedback
    return True, ""


def check_alerts(last_entry: Optional[dict] = None,
                 logs: Optional[list[dict]] = None) -> list[str]:
    """
    Check all alert thresholds after a query completes.

    Parameters
    ----------
    last_entry : The log entry just appended (for per-call checks).
    logs       : Full session log list (for rolling error-rate check).

    Returns
    -------
    List of human-readable warning strings (may be empty).
    Each string is suitable for passing directly to st.warning().
    """
    if logs is None:
        logs = _SESSION_LOGS

    alerts: list[str] = []

    # ── Per-call checks ────────────────────────────────────────────────────
    if last_entry:
        lat = last_entry.get("latency_seconds", 0)
        if lat > LATENCY_ALERT_THRESHOLD:
            alerts.append(
                f"🐢 **Latency alert:** last call took **{lat:.1f}s** "
                f"(threshold: {LATENCY_ALERT_THRESHOLD}s, "
                f"exceeded by {lat - LATENCY_ALERT_THRESHOLD:.1f}s)."
            )

        cost = last_entry.get("estimated_cost_usd", 0)
        if cost > COST_ALERT_THRESHOLD:
            alerts.append(
                f"💸 **Cost alert:** last query cost **${cost:.4f}** "
                f"(threshold: ${COST_ALERT_THRESHOLD:.2f}, "
                f"exceeded by ${cost - COST_ALERT_THRESHOLD:.4f})."
            )

    # ── Rolling error-rate check (last ERROR_RATE_WINDOW real calls) ───────
    real = _real_calls(logs)
    window = real[-ERROR_RATE_WINDOW:]  # up to 20; fewer if session is shorter
    if window:
        error_count = sum(1 for e in window if e.get("status") == "failure")
        rate = error_count / len(window)
        if rate > ERROR_RATE_ALERT_THRESHOLD:
            alerts.append(
                f"🚨 **Error-rate alert:** {error_count}/{len(window)} recent calls failed "
                f"({rate*100:.1f}% — threshold: {ERROR_RATE_ALERT_THRESHOLD*100:.0f}%)."
            )

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 4 — A/B grounding prompt variants
# ══════════════════════════════════════════════════════════════════════════════

# Version A — the original Day 4 grounding prompt addendum (baseline).
# The full SYSTEM_PROMPT template lives in rag_pipeline.py; Version A adds nothing.
PROMPT_VERSION_A_SUFFIX = ""  # no change to the baseline prompt

# Version B — adds strict citation and refusal rules on top of Version A.
PROMPT_VERSION_B_SUFFIX = (
    "\n\n════════════════════════════════════════\n"
    "ADDITIONAL GROUNDING RULES (Version B)\n"
    "════════════════════════════════════════\n"
    "• Cite [Section, Page] for EVERY fact you state. "
    "If a fact appears on a specific page of the knowledge base, reference it.\n"
    "• If the EXACT answer is not present verbatim in the provided context, "
    "you MUST respond: \"I don't have that specific information.\"\n"
    "• NEVER infer, extrapolate, or fill gaps from general knowledge "
    "— even if the answer seems obvious from context clues.\n"
    "• Borderline cases (partial info in context): state only what IS in context, "
    "then end with: \"I don't have the complete information — "
    "please contact BVRIT Hyderabad directly.\"\n"
)


def assign_prompt_version() -> str:
    """
    Randomly assign Version A or B for a single production query.
    Returns "A" or "B" with equal probability (50/50).
    Used for live traffic A/B testing only — controlled comparison tests
    should call get_prompt_suffix() with an explicit version argument.
    """
    return random.choice(["A", "B"])


def get_prompt_suffix(version: str) -> str:
    """
    Return the prompt suffix for the given version ("A" or "B").
    Version A returns an empty string (no change to baseline).
    Version B returns the additional strictness rules.
    """
    return PROMPT_VERSION_B_SUFFIX if version == "B" else PROMPT_VERSION_A_SUFFIX
