# Running Jarvis publicly (2026-11-26): an access code, signed sessions, a wrong-code limit and usage caps.
# Every setting is read from the environment at request time, so tests (and a restart) pick up changes.
#
#   JARVIS_ACCESS_CODE      set = the page is locked until the code is typed; unset (a laptop) = no lock
#   JARVIS_SESSION_SECRET   signs the session cookie (12 h). Unset: a random per-process secret (sessions end
#                           on restart), with a warning at start-up.
#   JARVIS_MAX_CALLS_PER_DAY (30), JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY (5), JARVIS_MAX_TURNS_PER_CALL (15)
#
# The code is only ever typed on the lock screen (POST body). It is never put in a URL, a page or a log, and it
# is compared in constant time. Wrong tries: 5 per browser session per 10 minutes, plus a ceiling for everyone
# together (30 per 10 minutes), because a forwarded IP header can't be trusted behind Tailscale Funnel (it adds
# no client identity, and a visitor could send their own X-Forwarded-For) and a session alone could be reset
# by clearing cookies.
import os
import hmac
import time
import hashlib
import secrets
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import db

SESSION_COOKIE = "jarvis_session"   # set after the right code: proves access, 12 h
CLIENT_COOKIE = "jarvis_client"     # set on the lock screen: who is trying (for the wrong-code limit)
SESSION_SECONDS = 12 * 3600
WRONG_PER_CLIENT, WRONG_PER_ALL, WRONG_WINDOW_S = 5, 30, 600
IST = timezone(timedelta(hours=5, minutes=30))

_fallback_secret = secrets.token_hex(32)
_lock = threading.Lock()
_wrong = {}          # client id -> deque of wrong-try times
_wrong_all = deque()  # every wrong try, any client


def access_code():
    return os.getenv("JARVIS_ACCESS_CODE") or ""


def is_locked():
    return bool(access_code())


def secret_is_set():
    return bool(os.getenv("JARVIS_SESSION_SECRET"))


def _signer(salt):
    return URLSafeTimedSerializer(os.getenv("JARVIS_SESSION_SECRET") or _fallback_secret, salt=salt)


def code_matches(given):
    """Constant time: compares fixed-length digests, so the time taken says nothing about the code."""
    given = given if isinstance(given, str) else ""
    return hmac.compare_digest(hashlib.sha256(given.encode("utf-8")).digest(),
                               hashlib.sha256(access_code().encode("utf-8")).digest())


def new_session():
    """(visitor id, signed cookie value) for a fresh session."""
    sid = secrets.token_urlsafe(16)
    return sid, _signer("jarvis-session").dumps({"sid": sid})


def session_visitor(cookie):
    """The visitor id from a valid, unexpired session cookie, else None."""
    if not cookie:
        return None
    try:
        return _signer("jarvis-session").loads(cookie, max_age=SESSION_SECONDS).get("sid")
    except (BadSignature, SignatureExpired, AttributeError):
        return None


def new_client():
    cid = secrets.token_urlsafe(16)
    return cid, _signer("jarvis-client").dumps({"cid": cid})


def client_id(cookie):
    if not cookie:
        return None
    try:
        return _signer("jarvis-client").loads(cookie).get("cid")
    except (BadSignature, AttributeError):
        return None


def _trim(times, now):
    while times and now - times[0] > WRONG_WINDOW_S:
        times.popleft()


def wrong_tries_wait(cid):
    """Seconds this client must wait before trying again (0 = may try now)."""
    now = time.time()
    with _lock:
        mine = _wrong.setdefault(cid, deque())
        _trim(mine, now)
        _trim(_wrong_all, now)
        waits = []
        if len(mine) >= WRONG_PER_CLIENT:
            waits.append(WRONG_WINDOW_S - (now - mine[0]))
        if len(_wrong_all) >= WRONG_PER_ALL:
            waits.append(WRONG_WINDOW_S - (now - _wrong_all[0]))
        return int(max(waits)) + 1 if waits else 0


def note_wrong_try(cid):
    now = time.time()
    with _lock:
        _wrong.setdefault(cid, deque()).append(now)
        _wrong_all.append(now)


def reset_wrong_tries():  # tests
    with _lock:
        _wrong.clear()
        _wrong_all.clear()


# ---------- usage caps (counts in jarvis.db, the day is India time) ----------

def _int_setting(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_calls_per_day():
    return _int_setting("JARVIS_MAX_CALLS_PER_DAY", 30)


def max_calls_per_visitor():
    return _int_setting("JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY", 5)


def max_turns_per_call():
    return _int_setting("JARVIS_MAX_TURNS_PER_CALL", 15)


def today():
    """Today's date in India time: the counts reset at midnight IST."""
    return datetime.now(IST).date().isoformat()


def _setup():
    conn = db.connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS usage (
        day TEXT NOT NULL, visitor TEXT NOT NULL, calls INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (day, visitor))""")
    conn.commit()
    conn.close()


def calls_today(visitor=None):
    _setup()
    conn = db.connect()
    if visitor is None:
        row = conn.execute("SELECT COALESCE(SUM(calls), 0) FROM usage WHERE day = ?", (today(),)).fetchone()
    else:
        row = conn.execute("SELECT COALESCE(SUM(calls), 0) FROM usage WHERE day = ? AND visitor = ?", (today(), visitor)).fetchone()
    conn.close()
    return row[0]


def call_cap_hit(visitor):
    """"day" or "visitor" if a new call would go over a cap, else None."""
    if calls_today() >= max_calls_per_day():
        return "day"
    if calls_today(visitor) >= max_calls_per_visitor():
        return "visitor"
    return None


def count_call(visitor):
    _setup()
    conn = db.connect()
    conn.execute("INSERT INTO usage (day, visitor, calls) VALUES (?, ?, 1) "
                 "ON CONFLICT(day, visitor) DO UPDATE SET calls = calls + 1", (today(), visitor))
    conn.commit()
    conn.close()


def demo_mode():
    return os.getenv("JARVIS_DEMO_MODE") == "1"


DEMO_NOTICE = "Calls are transcribed by third-party services and logged for testing; audio isn't stored."
