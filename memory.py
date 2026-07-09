"""
memory.py
---------
All five memory layers for the BVRIT Hyderabad RAG Chatbot.

Architecture overview
─────────────────────
Layer 1 — Short-term  : In-session message list (st.session_state.messages).
                        Clean history: user messages are stored WITHOUT the
                        retrieved-context block so history doesn't balloon.

Layer 2 — Medium-term : After every SUMMARIZE_EVERY turns, the oldest turns
                        (everything older than the most recent RECENT_TURNS turns)
                        are collapsed into a single summary turn via an LLM call.

Layer 3 — Long-term   : SQLite-backed user profile store (user_profiles.db).
                        Profile is loaded at session start and injected into the
                        system prompt. Updated at session end via an LLM extract call.

Layer 4 — Personalization : Per-user tone/format instructions derived from
                            the profile (detail_level, branch_interest, etc.).
                            Injected into the system prompt template.

Layer 5 — Privacy     : "clear my data" command, 30-day auto-expiry, first-run
                        privacy notice.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Observability: route all LLM calls through the logging wrapper ─────────────
from observability import logged_llm_call

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profiles.db")
SUMMARIZE_EVERY  = 10   # compress history after this many total turns
RECENT_TURNS     = 10   # keep this many recent turns verbatim after compression
PROFILE_TTL_DAYS = 30   # profiles not accessed in this many days are deleted
LLM_MODEL        = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", None)

# Privacy notice — shown on first interaction
PRIVACY_NOTICE = (
    "👋 **Welcome to the BVRIT Hyderabad Info Assistant!**\n\n"
    "To give you a personalised experience, I remember your name, branch interest, "
    "and conversation preferences across sessions. This information is stored locally "
    "on this server and is never shared with third parties. "
    "You can delete all your saved data at any time by typing **`clear my data`**."
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — Layer 3 (Long-term)
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the user_profiles table if it does not exist.
    Called once at app startup.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id             TEXT PRIMARY KEY,
                name                TEXT,
                branch_interest     TEXT,
                language_preference TEXT DEFAULT 'English',
                detail_level        TEXT DEFAULT 'detailed',
                topics_discussed    TEXT DEFAULT '[]',
                last_session_summary TEXT DEFAULT '',
                created_at          TEXT,
                last_accessed       TEXT
            )
        """)
        conn.commit()


def load_profile(user_id: str) -> dict:
    """
    Load profile for user_id from SQLite.
    Returns a dict with all profile fields (or defaults if new user).
    Updates last_accessed timestamp on load.

    Long-term memory: persists across process restarts.
    """
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            # New user — return empty defaults (do NOT insert yet)
            return {
                "user_id":             user_id,
                "name":                None,
                "branch_interest":     None,
                "language_preference": "English",
                "detail_level":        "detailed",
                "topics_discussed":    [],
                "last_session_summary": "",
                "created_at":          None,
                "last_accessed":       None,
                "is_new":              True,
            }
        # Update last_accessed
        conn.execute(
            "UPDATE user_profiles SET last_accessed = ? WHERE user_id = ?",
            (datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        profile = dict(row)
        profile["topics_discussed"] = json.loads(profile.get("topics_discussed") or "[]")
        profile["is_new"] = False
        return profile


def save_profile(profile: dict) -> None:
    """
    Upsert a user profile into SQLite.
    Creates a new row if the user_id does not exist, updates otherwise.

    Long-term memory: ensures data persists across sessions.
    """
    init_db()
    now = datetime.utcnow().isoformat()
    topics_json = json.dumps(profile.get("topics_discussed") or [])
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles
                (user_id, name, branch_interest, language_preference, detail_level,
                 topics_discussed, last_session_summary, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name                 = excluded.name,
                branch_interest      = excluded.branch_interest,
                language_preference  = excluded.language_preference,
                detail_level         = excluded.detail_level,
                topics_discussed     = excluded.topics_discussed,
                last_session_summary = excluded.last_session_summary,
                last_accessed        = excluded.last_accessed
        """, (
            profile["user_id"],
            profile.get("name"),
            profile.get("branch_interest"),
            profile.get("language_preference", "English"),
            profile.get("detail_level", "detailed"),
            topics_json,
            profile.get("last_session_summary", ""),
            profile.get("created_at") or now,
            now,
        ))
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Short-term memory helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_user_turn(content: str) -> dict:
    """
    Create a clean user turn for session history.
    The content should be the RAW user question — NO retrieved context attached.
    Context is injected per-turn inside run_rag_pipeline(), not stored here.
    """
    return {"role": "user", "content": content}


def make_assistant_turn(content: str, citations: list, refused: bool,
                        elapsed: float = 0.0, images: list = None) -> dict:
    """
    Create an assistant turn for session history.
    Stores the full response text so future turns can reference it for coreference.
    """
    return {
        "role":      "assistant",
        "content":   content,
        "citations": citations,
        "refused":   refused,
        "elapsed":   elapsed,
        "images":    images or [],
    }


def get_llm_history(messages: list[dict]) -> list[dict]:
    """
    Return the message list in the format expected by run_rag_pipeline's
    chat_history parameter — i.e. just role + content dicts.

    Short-term: the full in-session turn list is passed on every API call
    so the model can resolve references like 'the first one' or 'Priya'.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Medium-term memory (conversation summarization)
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm() -> ChatOpenAI:
    """Instantiate the LLM with project-standard settings."""
    kwargs = {"model": LLM_MODEL, "temperature": 0}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def _count_turns(messages: list[dict]) -> int:
    """Count the number of user turns (one 'turn' = one user + one assistant message)."""
    return sum(1 for m in messages if m["role"] == "user")


def _summarize_old_turns(old_turns: list[dict], llm: ChatOpenAI) -> str:
    """
    Summarize a list of (user, assistant) turn dicts into one paragraph.
    The summary preserves: user name, branches/topics discussed, specific
    fee amounts, dates, package figures, stated preferences, unresolved questions.

    Medium-term: replaces old turns to prevent token explosion.
    """
    convo_text = ""
    for m in old_turns:
        role = m["role"].capitalize()
        convo_text += f"{role}: {m['content'][:600]}\n"

    prompt = (
        "You are a conversation summarizer for a college chatbot.\n"
        "Summarize the conversation below into ONE paragraph of 5-8 sentences.\n\n"
        "You MUST preserve ALL of the following if they appear:\n"
        "- The user's name\n"
        "- Which branches/departments/programs were discussed\n"
        "- Specific numerical facts mentioned (exact fee amounts, placement packages, dates)\n"
        "- Any preferences the user stated (language, detail level, branch interest)\n"
        "- Any questions that were asked but not fully answered\n\n"
        "Do NOT add any new information. Summarize only what is present.\n\n"
        f"Conversation:\n{convo_text}\n\n"
        "Summary paragraph:"
    )
    try:
        resp = logged_llm_call(
            llm.invoke,
            "summarization",
            input=[HumanMessage(content=prompt)],
        )
        return resp.content.strip()
    except Exception as e:
        logger.warning(f"Summarization LLM call failed: {e}")
        # Fallback: concatenate user messages only
        return "Earlier conversation: " + " | ".join(
            m["content"][:80] for m in old_turns if m["role"] == "user"
        )


def maybe_summarize(messages: list[dict]) -> list[dict]:
    """
    Check if the conversation is long enough to warrant summarization.
    If total turns ≥ SUMMARIZE_EVERY, compress the oldest turns into a
    summary message and return the pruned message list.

    The returned list always keeps at least RECENT_TURNS verbatim turns
    plus one optional summary turn at the top.

    Medium-term memory: called after every assistant response.
    """
    turn_count = _count_turns(messages)
    if turn_count < SUMMARIZE_EVERY:
        return messages  # Not yet time to summarize

    # Separate old turns (to be summarized) from recent turns (kept verbatim)
    # We count backwards to find the cut-off index
    recent_user_turns_seen = 0
    cut_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            recent_user_turns_seen += 1
        if recent_user_turns_seen >= RECENT_TURNS:
            cut_idx = i
            break

    old_turns    = [m for m in messages[:cut_idx] if m["role"] in ("user", "assistant")]
    recent_turns = messages[cut_idx:]

    # Skip if the old turns are already a summary (role == "summary")
    already_summary = any(m.get("role") == "summary" for m in old_turns)

    if not old_turns or already_summary:
        return messages

    llm = _build_llm()
    summary_text = _summarize_old_turns(old_turns, llm)

    summary_turn = {
        "role":    "summary",
        "content": f"[Conversation summary up to this point]: {summary_text}",
    }
    return [summary_turn] + recent_turns


def get_llm_history_with_summary(messages: list[dict]) -> list[dict]:
    """
    Like get_llm_history() but treats summary turns as assistant context so
    the LLM sees the summary inline in the conversation history.
    """
    result = []
    for m in messages:
        role = m["role"]
        if role == "summary":
            # Inject as a system-like assistant message so the LLM sees it
            result.append({"role": "assistant", "content": m["content"]})
        elif role in ("user", "assistant"):
            result.append({"role": role, "content": m["content"]})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Long-term memory: profile update at session end
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_update_profile(messages: list[dict], profile: dict) -> dict:
    """
    Use the LLM to extract new facts from the current session and update
    the user profile dict in-place.

    Looks for: user name, branch interest, language preference, detail level,
    topics discussed, and a session summary.

    Long-term memory: called before save_profile() at session end or periodically.
    """
    if not messages:
        return profile

    # Build a compact conversation for the LLM to parse
    convo_text = ""
    for m in messages[-20:]:  # limit to last 20 turns to save tokens
        if m["role"] in ("user", "assistant"):
            convo_text += f"{m['role'].capitalize()}: {m['content'][:400]}\n"

    existing_facts = (
        f"Current profile:\n"
        f"  name={profile.get('name')}\n"
        f"  branch_interest={profile.get('branch_interest')}\n"
        f"  language_preference={profile.get('language_preference')}\n"
        f"  detail_level={profile.get('detail_level')}\n"
        f"  topics_discussed={profile.get('topics_discussed')}\n"
    )

    prompt = (
        "You are a user profile extractor for a college chatbot.\n"
        "Given the conversation and the existing profile, extract updated values.\n\n"
        f"{existing_facts}\n"
        "Conversation (most recent 20 turns):\n"
        f"{convo_text}\n\n"
        "Return a JSON object with ONLY fields that should be updated. "
        "Use null for fields you cannot determine. Fields:\n"
        "  name (string): user's name if stated\n"
        "  branch_interest (string): e.g. 'CSE', 'ECE', 'EEE', 'IT', 'CSE-AIML'\n"
        "  language_preference (string): 'English', 'Telugu', 'Hindi', etc.\n"
        "  detail_level (string): 'detailed' or 'brief'\n"
        "  new_topics (list of strings): any new topics discussed not already in profile\n"
        "  session_summary (string): one-paragraph summary of this session\n\n"
        "Respond with ONLY valid JSON, no markdown."
    )

    try:
        llm = _build_llm()
        resp = logged_llm_call(
            llm.invoke,
            "profile_extraction",
            input=[HumanMessage(content=prompt)],
        )
        raw = resp.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw)
    except Exception as e:
        logger.warning(f"Profile extraction failed: {e}")
        return profile

    # Apply extracted fields to profile (only if non-null)
    if extracted.get("name"):
        profile["name"] = extracted["name"]
    if extracted.get("branch_interest"):
        profile["branch_interest"] = extracted["branch_interest"]
    if extracted.get("language_preference"):
        profile["language_preference"] = extracted["language_preference"]
    if extracted.get("detail_level"):
        profile["detail_level"] = extracted["detail_level"]
    if extracted.get("session_summary"):
        profile["last_session_summary"] = extracted["session_summary"]

    # Merge new topics with existing (deduplicate)
    existing_topics = set(profile.get("topics_discussed") or [])
    new_topics = set(extracted.get("new_topics") or [])
    profile["topics_discussed"] = sorted(existing_topics | new_topics)

    return profile


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Personalization: system prompt prefix builder
# ══════════════════════════════════════════════════════════════════════════════

def build_personalization_prefix(profile: dict) -> str:
    """
    Build a personalization block that is prepended to the RAG system prompt.
    Uses profile fields to adapt tone, scope, and format.

    Personalization rules:
    - detail_level='detailed' → use full paragraphs with all available info
    - detail_level='brief'    → use short bullet-point answers
    - branch_interest set     → resolve "my branch" to that branch without asking
    - name set                → address the user by name in the first response

    Personalization: called on every turn with the current profile.
    """
    if not profile or profile.get("is_new"):
        return ""

    lines = ["════════════════════════════════════════",
             "USER PROFILE — PERSONALIZATION CONTEXT",
             "════════════════════════════════════════"]

    if profile.get("name"):
        lines.append(f"User name: {profile['name']} (address them by name occasionally, not on every reply)")

    if profile.get("branch_interest"):
        lines.append(
            f"Stored branch interest: {profile['branch_interest']}. "
            "When the user says 'my branch', always resolve this to "
            f"{profile['branch_interest']} WITHOUT asking them to restate it."
        )

    lang = profile.get("language_preference", "English")
    if lang and lang != "English":
        lines.append(f"Language preference: respond in {lang} where possible.")

    detail = profile.get("detail_level", "detailed")
    if detail == "brief":
        lines.append(
            "Response style: BRIEF. Use short bullet points. "
            "Give direct answers in 3-5 bullets. No long paragraphs."
        )
    else:
        lines.append(
            "Response style: DETAILED. Provide complete, well-structured paragraphs "
            "with all relevant facts from the context."
        )

    if profile.get("last_session_summary"):
        lines.append(f"Previous session summary: {profile['last_session_summary'][:300]}")

    if profile.get("topics_discussed"):
        topics_str = ", ".join(profile["topics_discussed"][:8])
        lines.append(f"Topics previously discussed: {topics_str}")

    lines.append("════════════════════════════════════════\n")
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Privacy: clear data, auto-expiry
# ══════════════════════════════════════════════════════════════════════════════

def delete_profile(user_id: str) -> bool:
    """
    Delete the user's profile row from SQLite.
    Returns True if a row was deleted, False if the user_id was not found.

    Privacy — right to be forgotten: honours 'clear my data' requests.
    """
    init_db()
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def expire_old_profiles(ttl_days: int = PROFILE_TTL_DAYS) -> int:
    """
    Delete profiles not accessed within ttl_days.
    Returns the number of profiles deleted.

    Privacy — auto-expiry: run at app startup.
    NOTE: In production, this should also run on a scheduled cron job,
    not only at startup. On-startup execution is a minimum viable approach.
    """
    init_db()
    cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM user_profiles WHERE last_accessed < ? OR last_accessed IS NULL",
            (cutoff,),
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info(f"Auto-expired {deleted} stale profile(s) (TTL={ttl_days} days).")
        return deleted


def is_clear_data_command(text: str) -> bool:
    """
    Detect if the user's message is a data-deletion request.
    Supports: 'clear my data', 'delete my data', 'forget me', 'erase my data'.
    """
    lower = text.lower().strip()
    triggers = [
        "clear my data", "delete my data", "erase my data",
        "forget me", "forget my data", "remove my data",
        "delete my profile", "clear my profile", "remove my profile",
        "reset my data", "wipe my data",
    ]
    return any(t in lower for t in triggers)


def handle_clear_data(user_id: str) -> str:
    """
    Delete the user's profile and return a confirmation message.
    Also removes the .user_id file so a fresh ID is created next session.
    Returns a human-readable string suitable for display in the chat UI.

    Privacy: called when is_clear_data_command() returns True.
    """
    # Remove the persisted user_id file so next session gets a fresh identity
    id_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".user_id")
    try:
        if os.path.exists(id_file):
            os.remove(id_file)
    except Exception:
        pass

    deleted = delete_profile(user_id)
    if deleted:
        return (
            "✅ **Your data has been cleared.**\n\n"
            "All saved information (name, branch interest, conversation history summary, "
            "and preferences) has been permanently deleted from our system. "
            "This session's conversation will also be reset. "
            "You are now being treated as a new user."
        )
    else:
        return (
            "ℹ️ **No saved data found for your session.**\n\n"
            "There was no stored profile associated with your session ID. "
            "Nothing has been deleted."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: user_id generator for Streamlit sessions
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_user_id(session_state) -> str:
    """
    Return a stable user_id that persists across browser sessions.

    Strategy: store the UUID in a local file (user_id.txt) so the same
    machine/user always gets the same ID regardless of browser refreshes.
    Falls back to a session-only UUID if the file cannot be written.

    In a real deployment this would be tied to an authenticated identity.
    """
    import uuid

    # 1. Already set in this Streamlit session — return immediately
    if "user_id" in session_state:
        return session_state["user_id"]

    # 2. Try to read a previously saved ID from disk
    id_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".user_id")
    try:
        if os.path.exists(id_file):
            with open(id_file, "r") as f:
                saved_id = f.read().strip()
            if saved_id:
                session_state["user_id"] = saved_id
                return saved_id
    except Exception:
        pass

    # 3. Generate a new UUID, persist it to disk for future sessions
    new_id = str(uuid.uuid4())
    try:
        with open(id_file, "w") as f:
            f.write(new_id)
    except Exception:
        pass  # if we can't write, the ID is still usable for this session

    session_state["user_id"] = new_id
    return new_id
