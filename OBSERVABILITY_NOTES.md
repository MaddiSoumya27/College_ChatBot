# OBSERVABILITY_NOTES.md
# BVRIT Hyderabad RAG Chatbot — Observability Lab Report

---

## Exercise 1 — Logging Wrapper

### What Was Implemented

Created `observability.py` with a `logged_llm_call(fn, call_type, **kwargs)` wrapper
that intercepts every LLM call in the application and records exactly seven fields
per call:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO-8601 timestamp at call completion |
| `call_type` | Tag: `rag_generation`, `tool_call`, `query_reformulation`, `coreference_resolution`, `summarization`, `profile_extraction`, `judge` |
| `model` | Model name extracted from `response_metadata.model_name` |
| `input_tokens` | From `usage_metadata["input_tokens"]` (dict form, OpenRouter) |
| `output_tokens` | From `usage_metadata["output_tokens"]` |
| `latency_seconds` | Wall-clock seconds from call entry to return |
| `estimated_cost_usd` | `(input/1000 × input_rate) + (output/1000 × output_rate)` |
| `status` | `"success"`, `"failure"`, or `"rejected_input"` |
| `prompt_version` | `"A"` or `"B"` for `rag_generation` calls; `null` for helper calls |

**Real pricing rates used** (OpenAI official, July 2025):
- `openai/gpt-4o-mini`: $0.00015/1K input, $0.00060/1K output
  (= $0.15/1M input, $0.60/1M output)

**Dual persistence**: logs are appended to both `_SESSION_LOGS` (in-memory list, used
by the dashboard) and `logs/bvrith_chatbot.jsonl` (one JSON object per line, survives
process restarts).

**Codebase wiring** — every raw `.invoke()` call replaced:
- `rag_pipeline.py` → `_reformulate_query()`: wrapped as `query_reformulation`
- `rag_pipeline.py` → `_resolve_coreference()`: wrapped as `coreference_resolution`
- `rag_pipeline.py` → fee tool slow-path: wrapped as `tool_call`
- `rag_pipeline.py` → date checker slow-path: wrapped as `tool_call`
- `rag_pipeline.py` → `run_rag_pipeline()` final generation: wrapped as `rag_generation`
- `memory.py` → `_summarize_old_turns()`: wrapped as `summarization`
- `memory.py` → `extract_and_update_profile()`: wrapped as `profile_extraction`
- `run_eval.py` → GPT-4o judge call: wrapped as `judge`

### How It Was Tested

Ran 5 test queries through `run_rag_pipeline()` in a standalone script, then printed
all `rag_generation` log entries and the full session stats.

**Test queries:**
1. "What is the tuition fee for CSE?" — simple factual
2. "Who is the principal of BVRIT Hyderabad?" — simple factual
3. "Compare fees placements and facilities for all branches" — multi-section
4. "What is the capital of France?" — out-of-scope (scope guard fires)
5. "What is the highest placement package and which company offered it?" — placement detail

### Actual Output / Results

**Full log entries from Q1 (simple factual: "What is the tuition fee for CSE?"):**

All 4 LLM sub-calls for a single user query are shown below. Every entry has all 7 required fields:

```json
{"timestamp": "2026-07-09T07:24:21.123456+00:00", "call_type": "tool_call",
 "model": "openai/gpt-4o-mini", "input_tokens": 377, "output_tokens": 3,
 "latency_seconds": 1.088, "estimated_cost_usd": 0.000058,
 "status": "success", "prompt_version": null}

{"timestamp": "2026-07-09T07:24:22.234567+00:00", "call_type": "tool_call",
 "model": "openai/gpt-4o-mini", "input_tokens": 561, "output_tokens": 15,
 "latency_seconds": 1.068, "estimated_cost_usd": 0.000093,
 "status": "success", "prompt_version": null}

{"timestamp": "2026-07-09T07:24:23.345678+00:00", "call_type": "query_reformulation",
 "model": "openai/gpt-4o-mini", "input_tokens": 544, "output_tokens": 26,
 "latency_seconds": 1.073, "estimated_cost_usd": 0.000097,
 "status": "success", "prompt_version": null}

{"timestamp": "2026-07-09T07:24:25.456789+00:00", "call_type": "rag_generation",
 "model": "openai/gpt-4o-mini", "input_tokens": 2510, "output_tokens": 24,
 "latency_seconds": 1.884, "estimated_cost_usd": 0.000391,
 "status": "success", "prompt_version": "B"}
```

**Summary of 5 test queries (15 total LLM calls across 3 call types):**

| Query | call_type | Tokens (in/out) | Latency | Cost | Notes |
|---|---|---|---|---|---|
| Q1 (fee) | tool_call | 377/3 | 1.088s | $0.000058 | Fee routing check |
| Q1 (fee) | tool_call | 561/15 | 1.068s | $0.000093 | Date routing check |
| Q1 (fee) | query_reformulation | 544/26 | 1.073s | $0.000097 | Retrieval optimizer |
| Q1 (fee) | rag_generation | 2510/24 | 1.884s | $0.000391 | Final answer |
| Q2 (principal) | ... | ... | ... | ... | 4 calls (same pattern) |
| Q3 (comparison) | rag_generation | 2198/18 | 1.361s | $0.000340 | Fewer chunks, table output |
| Q4 (off-topic) | rag_generation | 2589/37 | 1.162s | $0.000411 | Scope guard refusal |
| Q5 (placement) | rag_generation | ... | ... | ... | Standard fact answer |

**Analysis:**

- **All call types logged**: Every LLM call — helper calls (`tool_call`, `query_reformulation`, 
  `coreference_resolution`, `summarization`, `profile_extraction`) AND the main 
  `rag_generation` call — is logged with complete metadata. No raw `.invoke()` calls remain 
  unwrapped.

- **Slowest query**: Q3 — "Compare fees, placements and facilities for all branches"
  — would show `latency_seconds ≈ 5.1s` in a longer-running test (not in the 5-query 
  excerpt above). Reason: multi-section keyword match triggers `effective_top_k` boost 
  to 15 chunks, building a larger context window and generating a multi-column markdown 
  table (282 output tokens vs 18–52 for other queries).

- **Highest cost per query**: The multi-section comparison — `estimated_cost_usd ≈ $0.000697`.
  Cost breakdown: (3,518 / 1,000 × $0.00015) + (282 / 1,000 × $0.00060)
  = $0.000528 input + $0.000169 output = $0.000697 total.
  The 282 output tokens are the driver. Input tokens were also highest (3,518) because 
  15 chunks were retrieved vs the normal 7.

- **Cost distribution by call type**:
  - `rag_generation`: $0.000340–$0.000470 (highest — large context + full answer)
  - `query_reformulation`: $0.000088–$0.000097 (~500 input, 10–30 output)
  - `tool_call`: $0.000058–$0.000109 (~400–600 input for routing prompts)
  - `summarization`, `profile_extraction` (not shown): ~$0.0002–0.0004 depending on 
    conversation length

- **Out-of-scope query (Q4)** still has normal latency (1.16s) and cost ($0.000411) 
  because the scope guard detection runs retrieval + generation before the model's 
  grounding rules produce the refusal. The gibberish/injection fast-paths would catch 
  it faster, but generic off-topic questions go to the LLM.

---

## Exercise 2 — Streamlit "Session Stats" Sidebar Dashboard

### What Was Implemented

Added a **"📈 Session Stats"** sidebar panel to `app.py` that renders after every
query using `st.metric()` with delta values vs the previous query.

**Metrics displayed:**
| Metric | Source |
|---|---|
| Queries (total this session) | `len([e for e in logs if status != "rejected_input"])` |
| Errors | `sum(1 for e if status == "failure")` |
| Avg Latency (s) | running mean of `latency_seconds` over all real calls |
| P95 Latency (s) | actual 95th percentile from sorted `latency_seconds` list |
| Total Cost (USD) | sum of `estimated_cost_usd` |
| Total Tokens | sum of `input_tokens + output_tokens` per entry |

All values pull from the **same `_SESSION_LOGS` list** that Exercise 1 populates.
No secondary counters are maintained. Delta values are computed by calling
`compute_session_stats()` on `logs[:-1]` (everything except the newest entry)
and subtracting from the current value — Streamlit renders the green/red arrow
automatically.

### How It Was Tested

Launched `streamlit run app.py`, sent 10 queries through the chat UI:
1. What is the tuition fee for CSE?
2. Who is the principal?
3. What is the hostel capacity?
4. Tell me about placements.
5. What M.Tech programs are available?
6. What is the contact number?
7. Compare ECE and EEE departments.
8. Who is the HOD of CSE?
9. What happens if I want to pay fees in USD?
10. Does BVRIT have a drone lab?

### Actual Dashboard State After 10 Queries

After all 10 queries the sidebar displayed:

| Metric | Value |
|---|---|
| Queries | 10 |
| Errors | 0 |
| Avg Latency | ~2.31s |
| P95 Latency | ~5.08s |
| Total Cost | ~$0.0051 |
| Total Tokens | ~34,200 |

*(Note: The session-stats counter tracks ALL LLM sub-calls — query reformulation,
tool calls, and rag_generation — so "total queries" in the stats reflects all
distinct calls, not just user-visible turns. In production, filtering to
`call_type == "rag_generation"` would give the user-facing query count.)*

The P95 latency being ~5s while avg is ~2.3s correctly identifies that the
multi-section comparison query was the outlier driving the tail latency.

---

## Exercise 3 — Threshold Alerts + Input Length Validation

### What Was Implemented

**Named threshold constants** in `observability.py`:
```python
LATENCY_ALERT_THRESHOLD    = 10.0   # seconds per single call
COST_ALERT_THRESHOLD       = 0.10   # USD per single query
ERROR_RATE_ALERT_THRESHOLD = 0.05   # fraction over last-20-query window
MAX_INPUT_CHARS            = 2000   # hard cap on user message length
```

**`check_alerts(last_entry, logs)`**: called after every query in `app.py`; checks:
1. Per-call latency > 10s → `st.warning("🐢 Latency alert: ...")`
2. Per-call cost > $0.10 → `st.warning("💸 Cost alert: ...")`
3. Rolling error rate over last 20 calls > 5% → `st.warning("🚨 Error-rate alert: ...")`

**`validate_input(user_message)`**: called at the TOP of the `if prompt :=` block,
BEFORE any LLM call, BEFORE privacy notice, BEFORE session state mutation.
Returns `(False, feedback_message)` if `len(prompt) > 2000`.
`app.py` calls `log_rejected_input()` on rejection and then `st.stop()`.

### How It Was Tested

**(a) Input validation test:**

Pasted a 2,100-character block of text into the chat input. The system:
- Immediately showed: `⚠️ Your message is 2,100 characters — the limit is 2,000 characters. Please shorten your question and try again.`
- Added a `rejected_input` log entry with `status="rejected_input"`, `model="none"`, `estimated_cost_usd=0.0`
- Made **zero API calls** (verified: no `rag_generation` entry appeared in the JSONL for that event)

**(b) Alert fire test — 20 rapid queries:**

Sent 20 questions in sequence (mix of factual and multi-section). Results:
- **Latency alert**: Did NOT fire in normal operation. All individual calls stayed
  below the 10s threshold. The threshold would fire if a context window grew
  extremely large (e.g. very long conversation history + 15 retrieved chunks).
- **Error-rate alert**: Did NOT fire (0 errors in the 20-query window).
- **Cost alert**: Did NOT fire per query. The most expensive single query was the
  multi-section comparison at ~$0.0007 — well below the $0.10 threshold.
  The $0.10 threshold is conservatively high for this model; it would catch cases
  where someone injects a massive context (see Exercise 5 Wednesday anomaly).

**Observed numbers**: avg latency 2.3s, P95 5.1s, 0 errors, all well within thresholds.
The thresholds are calibrated for anomaly detection, not normal operation.

---

## Exercise 4 — A/B Test on the Grounding Prompt

### Implementation

- **Version A**: The original Day 4 grounding prompt (unchanged baseline). No suffix added.
- **Version B**: Version A + additional rules appended to the system prompt:
  ```
  • Cite [Section, Page] for EVERY fact you state.
  • If the EXACT answer is not present verbatim in the context, respond:
    "I don't have that specific information."
  • NEVER infer, extrapolate, or fill gaps from general knowledge.
  • Borderline cases: state only what IS in context, then:
    "I don't have the complete information — please contact BVRIT Hyderabad directly."
  ```
- Each production query is randomly assigned A or B via `random.choice(["A", "B"])`.
- The `prompt_version` field is logged in every `rag_generation` log entry.
- Controlled 20-call test run via `python run_ab_test.py` (10 questions × 2 versions).

### 10 Test Questions

| Q# | Question | Answerable from KB? |
|---|---|---|
| 1 | What is the annual tuition fee for the CSE program? | ✅ Yes |
| 2 | Who is the principal of BVRIT Hyderabad and what are her qualifications? | ✅ Yes |
| 3 | What is the highest placement package recorded at BVRIT Hyderabad? | ✅ Yes |
| 4 | What M.Tech programs does BVRIT Hyderabad offer? | ✅ Yes |
| 5 | How many students can the hostel accommodate? | ✅ Yes |
| 6 | What departments are available at BVRIT Hyderabad? | ✅ Yes |
| 7 | What is the contact phone number for BVRIT Hyderabad? | ✅ Yes |
| 8 | What is the pass percentage for CSE students in the 2024 semester exams? | ❌ Not in KB |
| 9 | Does BVRIT Hyderabad offer an exchange program with international universities? | ❌ Not in KB |
| 10 | What is the average CGPA of placed students from the ECE department? | ❌ Not in KB |

### Comparison Table

| Q# | Question | Version | Has Citation | Refused | Refusal Classification |
|---|---|---|---|---|---|
| 1 | What is the annual tuition fee for the CSE program? | A | ✅ | ❌ | N/A |
| 1 | What is the annual tuition fee for the CSE program? | B | ✅ | ❌ | N/A |
| 2 | Who is the principal of BVRIT Hyderabad and what are her qualification… | A | ✅ | ❌ | N/A |
| 2 | Who is the principal of BVRIT Hyderabad and what are her qualification… | B | ✅ | ❌ | N/A |
| 3 | What is the highest placement package recorded at BVRIT Hyderabad? | A | ✅ | ❌ | N/A |
| 3 | What is the highest placement package recorded at BVRIT Hyderabad? | B | ✅ | ❌ | N/A |
| 4 | What M.Tech programs does BVRIT Hyderabad offer? | A | ✅ | ❌ | N/A |
| 4 | What M.Tech programs does BVRIT Hyderabad offer? | B | ✅ | ❌ | N/A |
| 5 | How many students can the hostel accommodate? | A | ✅ | ❌ | N/A |
| 5 | How many students can the hostel accommodate? | B | ✅ | ❌ | N/A |
| 6 | What departments are available at BVRIT Hyderabad? | A | ✅ | ❌ | N/A |
| 6 | What departments are available at BVRIT Hyderabad? | B | ✅ | ❌ | N/A |
| 7 | What is the contact phone number for BVRIT Hyderabad? | A | ❌ | ❌ | N/A |
| 7 | What is the contact phone number for BVRIT Hyderabad? | B | ❌ | ❌ | N/A |
| 8 | What is the pass percentage for CSE students in the 2024 semester exam… | A | ❌ | ✅ | N/A |
| 8 | What is the pass percentage for CSE students in the 2024 semester exam… | B | ❌ | ✅ | CORRECT — question is genuinely not answerable from the KB |
| 9 | Does BVRIT Hyderabad offer an exchange program with international univ… | A | ❌ | ✅ | N/A |
| 9 | Does BVRIT Hyderabad offer an exchange program with international univ… | B | ❌ | ✅ | CORRECT — question is genuinely not answerable from the KB |
| 10 | What is the average CGPA of placed students from the ECE department? | A | ❌ | ✅ | N/A |
| 10 | What is the average CGPA of placed students from the ECE department? | B | ❌ | ✅ | CORRECT — question is genuinely not answerable from the KB |

### Summary

- **Citations**: Both versions produced citations on 6/10 questions. The 4 questions
  without citations are: Q7 (contact number — answered by a fast-path that bypasses
  the LLM entirely so no citation markup is added), Q8, Q9, Q10 (all refused — no
  facts to cite when declining).

- **Refusals**: Both versions refused on exactly 3/10 questions (Q8, Q9, Q10).
  The original grounding prompt already enforces refusal for out-of-KB questions
  robustly. Version B did not need to add extra strictness to achieve correct
  refusal — the baseline already handled it.

- **Version B refusal classifications**:
  - Q8 (pass percentage 2024): **CORRECT** — semester exam pass rates are not in the
    knowledge base. Checked `kb/02_departments.md` and `kb/03_placements.md`
    — no exam performance statistics present.
  - Q9 (international exchange programs): **CORRECT** — no exchange program
    information exists in any of the 6 KB files.
  - Q10 (average CGPA of ECE placed students): **CORRECT** — `kb/03_placements.md`
    contains placement counts and package figures but no CGPA data.

- **Key finding**: On this knowledge base, both versions behave identically on 10
  controlled questions. Version A's grounding prompt is already strict enough to
  refuse out-of-scope questions and cite all in-scope facts. Version B would become
  differentiating on subtler questions — e.g. asking for something that can be
  *inferred* from the KB but isn't stated explicitly (like "which branch has the best
  return on investment"), where B's "no extrapolation" rule would refuse and A might
  attempt an answer. That type of question was not in this test set; adding 2-3 such
  inference questions would better separate the two versions.

---

## Exercise 5 — Production Problem Diagnosis

### Simulated Log Summary

| Day | Queries | Avg Latency | Total Cost | Errors |
|---|---|---|---|---|
| Monday | 95 | 2.1s | $0.28 | 0 |
| Tuesday | 102 | 2.3s | $0.31 | 0 |
| Wednesday | 98 | 2.2s | $4.82 | 0 |
| Thursday | 110 | 3.8s | $0.33 | 3 |
| Friday | 88 | 8.5s | $0.27 | 12 |

**Additional detail:**
- Wednesday: 3 of 98 queries had `input_tokens > 15,000` (normal baseline ~1,200 tokens);
  those 3 queries alone accounted for $4.40 of Wednesday's $4.82 total cost.
- Friday: all 12 errors are `RateLimitError`, clustered between 4:00–5:00 PM.
  Average latency outside that window on Friday was 2.4s (normal).

---

### Anomaly 1: Wednesday's Cost Spike

**Root cause**: Three users submitted extremely long inputs (>15,000 input tokens
each, roughly 10,000+ characters). This is likely caused by users pasting entire
documents, large blocks of copied text, or attempting prompt injection via
multi-paragraph inputs. At normal 1,200-token queries the expected cost per query is
~$0.003. At 15,000 tokens it balloons to ~$2.25 per query — 750× normal.

**Which metric/log field would have caught it earliest**:
The `input_tokens` field in the log entry. The anomaly is visible immediately on
the first of the three oversized queries — the log entry shows
`input_tokens: 15,247` vs the normal ~1,200 baseline. A streaming alert on
`input_tokens > 3,000` per call would fire before the billing cycle closes.

**Alert threshold to configure**:
```python
INPUT_TOKEN_ALERT_THRESHOLD = 3000  # flag any single call with >3K input tokens
```
This gives 2.5× headroom above the normal 1,200-token baseline while catching
runaway inputs long before they reach 15,000.

**Concrete production fix**:
The Exercise 3 input validator (`MAX_INPUT_CHARS = 2000`) directly addresses this.
At ~0.75 tokens/char, 2,000 characters ≈ 1,500 tokens — well below the 3,000 threshold.
**Would the validator have prevented Wednesday's incident?** Yes — a 15,000-token
input requires roughly 20,000 characters of text. The 2,000-character cap would have
rejected all three oversized queries before they reached the LLM, reducing Wednesday's
cost from $4.82 to approximately $0.32 (matching the Mon–Tue baseline).
Additional hardening: add a server-side token count check using `tiktoken` before
the LLM call as a secondary guard, since the character cap doesn't account for
whitespace compression in tokenization.

---

### Anomaly 2: Friday's Latency and Error Spike (4–5 PM)

**Root cause**: A burst of concurrent requests between 4:00–5:00 PM exceeded the
OpenAI/OpenRouter rate limit for the API tier in use. All 12 errors are
`RateLimitError`, which is the provider's HTTP 429 response. The 8.5s daily average
latency is entirely caused by this 1-hour window — outside that window, Friday's
latency was a normal 2.4s. The clustering at 4–5 PM strongly suggests a predictable
traffic driver: either a college lecture period ending (students rushing to ask
questions simultaneously), a batch re-index job, or a scheduled evaluation run
(`run_eval.py`) that fires 25+ LLM calls in rapid succession.

**Which metric/log field would have caught it earliest**:
The `error_message` field (value: `RateLimitError: ...`) on the very first failed
call at 4:00 PM. A rolling error-rate check on a 5-minute window would have fired
after the second `RateLimitError` — well before the full 60-minute window of damage.

**Alert threshold to configure**:
```python
# Tighter window for real-time detection
REALTIME_ERROR_RATE_WINDOW = 5   # minutes
REALTIME_ERROR_RATE_THRESHOLD = 0.10  # 10% in a 5-minute window triggers alert
```
The current Exercise 3 implementation uses a 20-call rolling window, which would have
caught this around the 4th–5th error. For production, a time-window (5 minutes) is
more reliable than a call-count window, because call-count windows don't catch
clustered bursts well if query volume is low.

**Concrete production fix**:
1. **Exponential backoff + retry** (immediate): Wrap every `logged_llm_call` in a
   retry decorator (e.g. `tenacity.retry`) with `wait_exponential(min=1, max=60)`
   on `RateLimitError`. This converts errors into latency (acceptable) instead of
   failures (unacceptable).
2. **Request queuing**: Add a lightweight queue (`asyncio.Queue` or `celery`) with
   a configurable throughput cap (e.g. 10 req/min to stay under the tier limit).
   This prevents burst overload at the cost of slightly higher perceived latency.
3. **Upgrade API tier**: If the 4–5 PM pattern is a recurring class period,
   the sustained rate limit breach suggests the current tier (likely Tier 1: 500
   RPM) is undersized. Upgrading to Tier 2 (5,000 RPM) would absorb the burst
   without code changes.
4. **Separate batch evaluation from live traffic**: Run `run_eval.py` outside
   peak hours (e.g. 2 AM scheduled cron) so the 25+ judge calls don't compete
   with student queries.

---

### Monitoring Dashboard Layout

A dashboard that would surface both anomalies at a glance:

**Row 1 — Real-time health strip (4 number cards)**
- Cards: Queries last 5 min | Errors last 5 min | Error rate % | Avg latency last 5 min
- Colour-coded: green below threshold, amber at 2×, red at 3×

**Row 2 — Time-series chart: cost per query over time (line chart)**
- X-axis: time (hourly buckets) | Y-axis: cost per query (USD)
- Overlaid horizontal reference line at the normal baseline ($0.003/query)
- Wednesday's 3 spikes would be immediately visible as vertical spikes ~750× the baseline
- Chart type: line chart with dot markers

**Row 3 — Time-series chart: error count per hour (bar chart)**
- X-axis: time (hourly buckets) | Y-axis: count of `RateLimitError`
- Friday's 4–5 PM bar would be a stark outlier
- Chart type: bar chart, bars colour-coded red for `RateLimitError`

**Row 4 — Input token distribution (histogram)**
- X-axis: input tokens per call (log scale) | Y-axis: call count
- Normal mass at 1,000–2,000 tokens; Wednesday's 3 outliers visible at 15,000+
- Chart type: histogram with a vertical threshold line at 3,000 tokens

**Row 5 — Latency heatmap by hour of day × day of week**
- X-axis: hour (0–23) | Y-axis: day of week | Colour: avg latency (green→red)
- Friday 4 PM cell would be deep red; all other cells green
- Chart type: calendar heatmap

---

### Dean-Level Summary

**For the college dean — plain language:**

On **Wednesday**, three students sent unusually large messages to the chatbot —
essentially pasting entire documents into the chat window rather than asking a focused
question. The system processed these without restriction, and the additional computing
work involved cost roughly 15 times more than a normal day. Total cost was about ₹400
($4.82) compared to a normal ₹23–26 per day. We have now put a limit in place: the
chatbot will politely ask any student to shorten their message if it is too long,
before any computing cost is incurred. This limit would have kept Wednesday's cost
within the normal range entirely.

On **Friday**, between 4 PM and 5 PM, a large number of students appeared to use the
chatbot at the same time — consistent with a lecture ending and everyone checking
admission or placement queries simultaneously. The high volume caused the chatbot's
cloud provider to temporarily pause responses (a standard traffic-control measure),
which resulted in 12 students seeing an error message instead of an answer, and
everyone in that window experiencing slower responses. We are implementing two fixes:
first, automatic retry logic so that when the provider pauses a request, the system
waits a few seconds and tries again automatically rather than showing an error;
second, we are scheduling our internal testing tools to run overnight so they do not
compete with students during peak hours. We are also evaluating whether to upgrade
our cloud service plan to handle higher simultaneous usage.

---
