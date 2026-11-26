# WhatsApp notifications via Jatin's n8n "Jarvis notes" webhook (2026-11-26, phase 3).
#
# After a reminder or a message is saved, the note goes into an OUTBOX table in jarvis.db first, then a background
# worker POSTs it. The call never waits for the webhook: saving the outbox row is a local SQLite write, and the
# POST happens on another thread. A failed POST stays in the outbox and is retried with backoff, also after a
# server restart (start() picks up whatever is still pending).
#
# The note_id is made once, when the note is saved, and stored with it: every retry sends the SAME id, so n8n can
# drop a repeat (it answers "duplicate": true, which counts as delivered).
#
# Off unless N8N_JARVIS_WEBHOOK_URL is set. The secret (N8N_JARVIS_SECRET) goes only in the X-Jarvis-Secret header:
# never logged, never printed, never in an error message.
import os
import json
import time
import threading
import sqlite3
import httpx
import db

TIMEOUT_S = 10
BACKOFF_S = [10, 30, 120, 600, 1800, 3600]  # then every hour
MAX_ATTEMPTS = 48                            # about two days, then it's marked given up (logged)

_wake = threading.Event()  # set by enqueue; an Event, so a wake-up during a delivery is never lost
_worker = {"thread": None, "stop": False}


def url():
    return (os.getenv("N8N_JARVIS_WEBHOOK_URL") or "").strip()


def enabled():
    return bool(url())


def setup():
    conn = db.connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outbox (
            note_id    TEXT PRIMARY KEY,
            call_id    INTEGER,
            payload    TEXT NOT NULL,
            attempts   INTEGER NOT NULL DEFAULT 0,
            next_try   REAL NOT NULL,
            last_error TEXT,
            status     TEXT NOT NULL DEFAULT 'pending',   -- pending / sent / gave_up
            created_at TEXT NOT NULL,
            sent_at    TEXT
        )
    """)
    conn.commit()
    conn.close()


def summary_line(kind, caller_name, text, urgency, is_family=False):
    """One line for the WhatsApp message, built by code from the saved note (no LLM rewriting).
    Family (Jatin, 2026-11-26): "🏠 Family call – Papa left a message: ..." / "🏠 Family call – Papa wants a callback (urgent)"."""
    text = " ".join((text or "").split())
    if len(text) > 140:
        text = text[:137].rstrip() + "..."
    urgent = " (urgent)" if urgency == "high" else ""
    if is_family:
        if kind == "reminder":
            about = "" if text.lower() in ("", "wants a callback") else f": {text}"
            return f"🏠 Family call – {caller_name} wants a callback{about}{urgent}"
        return f"🏠 Family call – {caller_name} left a message: {text}"
    if kind == "reminder":
        what = "wants a callback" if text.lower() in ("", "wants a callback") else f"{text}, wants a callback"
        return f"📞 {caller_name} called: {what}{urgent}"
    return f"📝 {caller_name} left a message: {text}"


def build_note(call_id, kind, n, caller_name, is_family, text, urgency=None):
    return {
        "note_id": f"jarvis-{call_id}-{kind}-{n}",
        "type": kind,
        "call_id": int(call_id),
        "caller_name": caller_name,
        "is_family": bool(is_family),
        "urgency": urgency if kind == "reminder" and urgency in ("low", "medium", "high") else None,
        "text": text,
        # Demo mode (JARVIS_DEMO_MODE=1): every note from a public demo call is marked, so it's never mistaken for a real one.
        "summary": ("🧪 Demo call: " if os.getenv("JARVIS_DEMO_MODE") == "1" else "")
                   + summary_line(kind, caller_name, text, urgency, is_family),
    }


def enqueue(note):
    """Store the note in the outbox and wake the worker. Returns at once. Does nothing if notifications are off."""
    if not enabled():
        return False
    setup()
    conn = db.connect()
    conn.execute("INSERT OR IGNORE INTO outbox (note_id, call_id, payload, next_try, created_at) VALUES (?, ?, ?, ?, ?)",
                 (note["note_id"], note["call_id"], json.dumps(note, ensure_ascii=False), time.time(), db.now()))
    conn.commit()
    conn.close()
    start()
    _wake.set()
    return True


def post(note):
    """One delivery attempt. Returns None on success, else the reason: "401", "400", "5xx (502)", "timeout",
    "unreachable", or "bad reply" (a 200 that doesn't say saved). Never includes the secret."""
    headers = {"X-Jarvis-Secret": os.getenv("N8N_JARVIS_SECRET") or "", "Content-Type": "application/json"}
    try:
        response = httpx.post(url(), content=json.dumps(note, ensure_ascii=False).encode("utf-8"),
                              headers=headers, timeout=TIMEOUT_S)
    except httpx.TimeoutException:
        return "timeout"
    except httpx.HTTPError:
        return "unreachable"
    if response.status_code >= 500:
        return f"5xx ({response.status_code})"
    if response.status_code != 200:
        return str(response.status_code)
    try:
        body = response.json()
    except ValueError:
        return "bad reply"
    if isinstance(body, list) and body:  # n8n can wrap a response in a list
        body = body[0]
    if isinstance(body, dict) and (body.get("saved") is True or body.get("duplicate") is True):
        return None
    return "bad reply"


def backoff(attempts):
    return BACKOFF_S[min(attempts - 1, len(BACKOFF_S) - 1)]


def deliver_due():
    """Send every pending note that is due. Returns seconds until the next one is due (None if nothing pending)."""
    setup()
    conn = db.connect()
    rows = conn.execute("SELECT * FROM outbox WHERE status = 'pending' ORDER BY next_try").fetchall()
    conn.close()
    now = time.time()
    next_due = None
    for row in rows:
        if row["next_try"] > now:
            next_due = row["next_try"] - now if next_due is None else min(next_due, row["next_try"] - now)
            continue
        note = json.loads(row["payload"])
        attempts = row["attempts"] + 1
        reason = post(note) if enabled() else "notifications off (no URL)"
        conn = db.connect()
        if reason is None:
            conn.execute("UPDATE outbox SET status = 'sent', attempts = ?, sent_at = ?, last_error = NULL WHERE note_id = ?",
                         (attempts, db.now(), row["note_id"]))
            conn.commit()
            conn.close()
            db.log(row["call_id"], "notify_sent", f"{row['note_id']} (attempt {attempts})")
            continue
        if attempts >= MAX_ATTEMPTS:
            conn.execute("UPDATE outbox SET status = 'gave_up', attempts = ?, last_error = ? WHERE note_id = ?",
                         (attempts, reason, row["note_id"]))
            conn.commit()
            conn.close()
            db.log(row["call_id"], "notify_failed", f"{row['note_id']}: {reason}, attempt {attempts}, gave up")
            continue
        wait = backoff(attempts)
        conn.execute("UPDATE outbox SET attempts = ?, last_error = ?, next_try = ? WHERE note_id = ?",
                     (attempts, reason, now + wait, row["note_id"]))
        conn.commit()
        conn.close()
        db.log(row["call_id"], "notify_failed", f"{row['note_id']}: {reason}, attempt {attempts}, retry in {wait}s")
        next_due = wait if next_due is None else min(next_due, wait)
    return next_due


def _run():
    while not _worker["stop"]:
        _wake.clear()
        try:
            wait = deliver_due()
        except Exception as error:  # never let the worker die: log nothing secret, try again later
            print(f"[notify] worker error: {type(error).__name__}")
            wait = 60
        _wake.wait(timeout=wait if wait is not None else 3600)


def start():
    """Start the background worker (once). server.py calls it at start-up, so notes left over are retried."""
    if not enabled():
        return False
    if _worker["thread"] is None or not _worker["thread"].is_alive():
        _worker["stop"] = False
        _worker["thread"] = threading.Thread(target=_run, name="notify", daemon=True)
        _worker["thread"].start()
    return True


def pending(call_id=None):
    """Outbox rows not yet delivered (for show.py)."""
    setup()
    conn = db.connect()
    if call_id is None:
        rows = conn.execute("SELECT * FROM outbox WHERE status != 'sent' ORDER BY created_at").fetchall()
    else:
        rows = conn.execute("SELECT * FROM outbox WHERE status != 'sent' AND call_id = ? ORDER BY created_at", (call_id,)).fetchall()
    conn.close()
    return rows
