# Public-run tests: access code, sessions, wrong-code limit, usage caps, demo mode, base path. No paid API calls:
# every paid client is replaced by a tripwire that records any use.
#   python test_access.py
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.DB_FILE = os.path.join(tempfile.gettempdir(), "jarvis_access_test.db")
if os.path.exists(db.DB_FILE):
    os.remove(db.DB_FILE)

db.setup()
import agent
for _name in ("JARVIS_ACCESS_CODE", "JARVIS_DEMO_MODE", "JARVIS_BASE_PATH", "JARVIS_SESSION_SECRET", "FAMILY_NAMES"):
    os.environ.pop(_name, None)
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="1000", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="1000", JARVIS_MAX_TURNS_PER_CALL="1000")
import access
import notify
import server
import voice

results = []


def check(name, condition):
    results.append(bool(condition))
    print(("  ok   " if condition else "  FAIL ") + name)


# ---------- tripwires: any paid request is recorded ----------
paid = []


def tripwire(name):
    def call(*args, **kwargs):
        paid.append(name)
        raise AssertionError(f"paid service called: {name}")
    return call


agent.client.chat.completions.create = tripwire("openai")
agent.main_client.chat.completions.create = tripwire("openai-main")
voice.transcribe = tripwire("stt")
voice.open_stream = tripwire("tts")
voice.warm_tts = lambda: None
voice.warm_stt = lambda: None
voice.close_tts = lambda: None
voice.cached_pcm = lambda text, lang: b"\x01\x00" * 480   # every fixed line "cached": no TTS
agent.RING_SECONDS = 0.2
CODE = "orange-kite-2741"
HTTPS = "https://localhost"


def fresh_client():
    return server.app.test_client()


def get(c, path, **kw):
    return c.get(path, base_url=HTTPS, **kw)


def post(c, path, body=None, **kw):
    return c.post(path, json=body or {}, base_url=HTTPS, **kw)


def cookie_header(response, name):
    return next((h for h in response.headers.getlist("Set-Cookie") if h.startswith(name + "=")), "")


def unlocked_client():
    c = fresh_client()
    get(c, "/session")
    assert post(c, "/unlock", {"code": CODE}).status_code == 200
    return c


def hang_up_all():
    server.active["call"] = False


print("lock off (no JARVIS_ACCESS_CODE): everything as before")
c = fresh_client()
check("/session: not locked", get(c, "/session").get_json()["locked"] is False)
check("/caller_ids works without a session", get(c, "/caller_ids").status_code == 200)
page = get(c, "/").get_data(as_text=True)
check("page has <base href=\"/\"> and the lock screen markup (hidden)", '<base href="/">' in page and 'id="lock"' in page)

print("lock on: page and lock screen open, everything else 401 before any paid request")
os.environ["JARVIS_ACCESS_CODE"] = CODE
os.environ["JARVIS_SESSION_SECRET"] = "test-secret-not-real"
c = fresh_client()
check("page itself is served (it shows the lock screen)", get(c, "/").status_code == 200)
r = get(c, "/session")
client_cookie = cookie_header(r, access.CLIENT_COOKIE)
check("/session: locked, and a client cookie (HttpOnly, Secure, SameSite=Lax)", r.get_json()["locked"] is True
      and "HttpOnly" in client_cookie and "Secure" in client_cookie and "SameSite=Lax" in client_cookie)
check("static files stay open", get(c, "/static/record.html").status_code == 200)
guarded = []
for rule in server.app.url_map.iter_rules():
    if rule.endpoint in server.OPEN_ENDPOINTS:
        continue
    path = rule.rule.replace("<token>", "abc")
    for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
        response = c.open(path, method=method, json={"text": "hi", "caller_id": "", "code": CODE, "answer": "accept"},
                          base_url=HTTPS)
        guarded.append((method, path, response.status_code))
check(f"every other endpoint refuses with 401 without a session ({len(guarded)}: "
      + ", ".join(f"{m} {p}" for m, p, _ in guarded) + ")",
      guarded and all(code == 401 for _, _, code in guarded))
check("covers start, turn, turn_audio, speech, filler, caller_ids, ring, answer, barge_in, hangup, client_timing",
      {p for _, p, _ in guarded} >= {"/start", "/turn", "/turn_audio", "/speech/abc", "/filler", "/caller_ids", "/ring",
                                     "/answer", "/barge_in", "/hangup", "/client_timing"})
r = c.post("/turn_audio", data={"audio": (__import__("io").BytesIO(b"x" * 100), "t.webm")}, base_url=HTTPS)
check("a spoken turn without a session: 401, no transcription", r.status_code == 401 and not paid)
check("no paid service was touched", paid == [])

print("unlock: wrong code, right code, session cookie")
c = fresh_client()
check("unlock with no client cookie (no lock screen seen): refused", post(c, "/unlock", {"code": CODE}).status_code == 400)
get(c, "/session")
check("wrong code: 401, still locked", post(c, "/unlock", {"code": "nope"}).status_code == 401
      and get(c, "/session").get_json()["locked"] is True)
r = post(c, "/unlock", {"code": CODE})
session_cookie = cookie_header(r, access.SESSION_COOKIE)
check("right code: 200 and a signed session cookie, HttpOnly, Secure, SameSite=Lax, 12 h",
      r.status_code == 200 and "HttpOnly" in session_cookie and "Secure" in session_cookie and "SameSite=Lax" in session_cookie
      and "Max-Age=43200" in session_cookie)
check("unlocked: /session says so, /caller_ids works", get(c, "/session").get_json()["locked"] is False
      and get(c, "/caller_ids").status_code == 200)
check("the code is not in the cookie", CODE not in session_cookie)
forged = fresh_client()
forged.set_cookie(access.SESSION_COOKIE, access.URLSafeTimedSerializer("wrong-secret", salt="jarvis-session").dumps({"sid": "x"}),
                  domain="localhost", secure=True)
check("a cookie signed with another secret: 401", get(forged, "/caller_ids").status_code == 401)
old_seconds, access.SESSION_SECONDS = access.SESSION_SECONDS, -1
check("an expired session: 401", get(c, "/caller_ids").status_code == 401)
access.SESSION_SECONDS = old_seconds
check("?code= in the URL is ignored: still locked", get(fresh_client(), f"/?code={CODE}").status_code == 200
      and get(fresh_client(), "/session").get_json()["locked"] is True)
src = open("access.py", encoding="utf-8").read()
check("constant-time comparison (hmac.compare_digest on digests)", "hmac.compare_digest(hashlib.sha256" in src)

print("wrong-code limit: 5 per browser per 10 minutes, 30 for everyone")
access.reset_wrong_tries()
c = fresh_client()
get(c, "/session")
codes = [post(c, "/unlock", {"code": f"wrong{i}"}).status_code for i in range(5)]
r = post(c, "/unlock", {"code": CODE})
check("5 wrong tries, then even the right code waits (429 with a wait)", codes == [401] * 5 and r.status_code == 429
      and r.get_json()["wait"] > 500 and "Too many tries" in r.get_json()["error"])
other = fresh_client()
get(other, "/session")
check("another browser can still try", post(other, "/unlock", {"code": CODE}).status_code == 200)
access.reset_wrong_tries()
for i in range(6):   # 6 browsers x 5 wrong = 30
    b = fresh_client()
    get(b, "/session")
    for j in range(5):
        post(b, "/unlock", {"code": "x"})
fresh = fresh_client()
get(fresh, "/session")
check("30 wrong tries in all: a brand-new browser waits too (clearing cookies doesn't help)",
      post(fresh, "/unlock", {"code": CODE}).status_code == 429)
access.reset_wrong_tries()
conn = db.connect()
everything = " ".join(str(v) for row in conn.execute("SELECT * FROM events") for v in row)
conn.close()
check("the code and the secret are in no log row", CODE not in everything and "test-secret-not-real" not in everything)
check("the code is in no page or link", CODE not in get(fresh_client(), "/").get_data(as_text=True))

print("usage caps: calls per day, per visitor per day, turns per call")
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="3", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="2")
started_calls = []
real_start_call = agent.start_call
agent.start_call = lambda **kw: (started_calls.append(1), real_start_call(**kw))[1]
access.today = lambda: "2026-12-01"
a, b2 = unlocked_client(), unlocked_client()
ra = [post(a, "/start", {"caller_id": ""}) for _ in range(3)]
hang_up_all()
check("visitor cap: a visitor's 3rd call that day is refused (429), the call never starts",
      [r.status_code for r in ra] == [200, 200, 429] and len(started_calls) == 2
      and ra[2].get_json()["error"].startswith("Demo limit reached for today"))
rb = [post(b2, "/start", {"caller_id": ""}) for _ in range(2)]
hang_up_all()
check("daily cap (3 for everyone): another visitor gets 1 more, then 429", [r.status_code for r in rb] == [200, 429]
      and len(started_calls) == 3 and rb[1].get_json()["limit"] == "day")
access.today = lambda: "2026-12-02"
check("a new day (midnight IST) resets the counts", post(b2, "/start", {"caller_id": ""}).status_code == 200)
hang_up_all()
check("the day is India time", access.IST.utcoffset(None).total_seconds() == 5.5 * 3600)
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="1000", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="1000", JARVIS_MAX_TURNS_PER_CALL="3")
c = unlocked_client()
post(c, "/start", {"caller_id": ""})
agent.call_state["turns"] = 3
paid.clear()
r = post(c, "/turn", {"text": "hello again"})
data = r.get_json()
check("turn cap: the next typed turn gets the fixed limit line, the call ends, page message, no model request",
      data["reply"] == server.LIMIT_LINES["english"] and data["ended"] is True and data["limit"] == "turns"
      and "Demo limit" in data["message"] and paid == [] and server.active["call"] is False)
post(c, "/start", {"caller_id": ""})
agent.call_state["turns"] = 3
r = c.post("/turn_audio", data={"audio": (__import__("io").BytesIO(b"x" * 100), "t.webm")}, base_url=HTTPS)
check("turn cap on a spoken turn: no transcription, the limit line", r.get_json()["ended"] is True and paid == []
      and r.get_json()["reply"] == server.LIMIT_LINES["english"])
check("the limit line is cached (no live speech): in the fixed-line set",
      all(agent.pick_line(server.LIMIT_LINES, l) in server.server_fixed_lines()[l] for l in ("english", "hinglish")))
check("settings default to 30 / 5 / 15", [access._int_setting(n, d) for n, d in (("NOPE_A", 30), ("NOPE_B", 5), ("NOPE_C", 15))]
      == [30, 5, 15] and "JARVIS_MAX_CALLS_PER_DAY\", 30" in src and "PER_VISITOR_PER_DAY\", 5" in src
      and "JARVIS_MAX_TURNS_PER_CALL\", 15" in src)
os.environ["JARVIS_MAX_TURNS_PER_CALL"] = "1000"
agent.start_call = real_start_call

print("demo mode: family names hidden, notice, 🧪 notes")
os.environ.update(JARVIS_DEMO_MODE="1", FAMILY_NAMES="1111:Papa,2222:Mummy")
c = unlocked_client()
labels = {i["id"]: i["label"] for i in get(c, "/caller_ids").get_json()["family"]}
check("/caller_ids: 'Family (1111)', never the names", labels == {"1111": "Family (1111)", "2222": "Family (2222)", "3333": "Family (3333)"})
s = get(fresh_client(), "/session").get_json()
check("lock screen notice", s["demo"] is True and s["notice"] == "Calls are transcribed by third-party services and logged for testing; audio isn't stored.")
check("no name in the page", "Papa" not in get(c, "/").get_data(as_text=True))
post(c, "/start", {"caller_id": "1111"})   # rings, no answer in 0.2 s: a family call that goes on
briefings = [r["detail"] for r in db.replay(agent.call_state["call_id"]) if r["kind"] == "owner_briefed"]
check("Jatin's phone panel (shown on the page): no name", briefings == ["Family call (caller ID 1111)"])
result = agent.take_message("Ghar jaldi aa jao.")
saved = db.reminders_for_call(agent.call_state["call_id"])[0]
check("the tool result the model reads: 'Family', not the name", "saved from Family." in result and "Papa" not in result)
check("the saved note keeps the real name (private, for WhatsApp)", saved["caller_name"] == "Papa")
note = notify.build_note(agent.call_state["call_id"], "message", 1, "Papa", True, "Ghar jaldi aa jao.")
check("WhatsApp summary starts '🧪 Demo call: ' and keeps the mapped name",
      note["summary"] == "🧪 Demo call: 🏠 Family call – Papa left a message: Ghar jaldi aa jao.")
check("normal caller in demo mode: '🧪 Demo call: 📞 Rahul called: ...'",
      notify.build_note(9, "reminder", 1, "Rahul", False, "VPS", "high")["summary"] == "🧪 Demo call: 📞 Rahul called: VPS, wants a callback (urgent)")
hang_up_all()
os.environ.pop("JARVIS_DEMO_MODE")
check("demo off: no prefix, names shown again", not notify.build_note(9, "message", 1, "Papa", True, "x")["summary"].startswith("🧪")
      and agent.caller_label("1111") == "Papa (1111)")
os.environ.pop("FAMILY_NAMES")

print("base path: /jarvis")
os.environ["JARVIS_BASE_PATH"] = "/jarvis"
c = fresh_client()
r = get(c, "/jarvis/")
check("page under /jarvis/ with <base href=\"/jarvis/\">", r.status_code == 200 and '<base href="/jarvis/">' in r.get_data(as_text=True))
check("/jarvis (no slash) works too", get(c, "/jarvis").status_code == 200)
r = get(c, "/jarvis/session")
check("cookies are scoped to the base path", "Path=/jarvis" in cookie_header(r, access.CLIENT_COOKIE))
check("unlock under the base path", post(c, "/jarvis/unlock", {"code": CODE}).status_code == 200)
check("an endpoint under /jarvis works with the session", get(c, "/jarvis/caller_ids").status_code == 200)
check("a static file under /jarvis", get(c, "/jarvis/static/record.html").status_code == 200)
# A proxy that strips the prefix: the browser still asked for /jarvis/... (so its cookie was sent); the server sees
# /caller_ids. The test client filters cookies by path, so this routing check runs with the lock off.
os.environ.pop("JARVIS_ACCESS_CODE")
check("a proxy that strips the prefix: /caller_ids and / still reach their routes",
      get(fresh_client(), "/caller_ids").status_code == 200 and '<base href="/jarvis/">' in get(fresh_client(), "/").get_data(as_text=True))
os.environ["JARVIS_ACCESS_CODE"] = CODE
r = post(c, "/jarvis/start", {"caller_id": ""})
url = r.get_json()["audio_url"]
check("audio URLs are relative (resolved against <base href>) and play under /jarvis",
      url.startswith("speech/") and get(c, "/jarvis/" + url).status_code == 200)
hang_up_all()
page = open("static/index.html", encoding="utf-8").read()
import re
check("the page uses no absolute URL (every fetch follows the base path)",
      not re.search(r"""(fetch|post|queueLine|sendTurn)\(\s*["'`]/""", page) and 'path === "turn_audio"' in page)
os.environ.pop("JARVIS_BASE_PATH")

print("host binding and settings")
main_src = open("server.py", encoding="utf-8").read()
check("JARVIS_HOST defaults to 127.0.0.1, JARVIS_PORT to 5000", 'os.getenv("JARVIS_HOST") or "127.0.0.1"' in main_src
      and 'os.getenv("JARVIS_PORT") or "5000"' in main_src and "app.run(host=host, port=port" in main_src)
example = open(".env.example", encoding="utf-8").read()
for name in ("JARVIS_ACCESS_CODE", "JARVIS_SESSION_SECRET", "JARVIS_MAX_CALLS_PER_DAY", "JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY",
             "JARVIS_MAX_TURNS_PER_CALL", "JARVIS_DEMO_MODE", "JARVIS_BASE_PATH", "JARVIS_HOST", "JARVIS_PORT"):
    check(f".env.example lists {name} (no real value)", re.search(rf"^#?\s*{name}=", example, re.M) is not None
          and not re.search(rf"^{name}=\S", example.replace("JARVIS_HOST=127.0.0.1", "").replace("JARVIS_PORT=5000", ""), re.M))
check("no paid service was touched in this whole file", paid == [])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
