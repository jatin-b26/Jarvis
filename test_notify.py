# Phase 3 tests: WhatsApp notes to the n8n webhook, against a local STUB webhook (never the real one).
#   python test_notify.py
import os
import sys
import json
import time
import tempfile
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.DB_FILE = os.path.join(tempfile.gettempdir(), "jarvis_notify_test.db")
if os.path.exists(db.DB_FILE):
    os.remove(db.DB_FILE)

import agent
# Public-run settings (access.py) pinned for the tests, whatever .env says: no lock, no demo mode, no base path,
# caps high enough for the many calls a suite starts. The access tests set their own values.
for _name in ("JARVIS_ACCESS_CODE", "JARVIS_DEMO_MODE", "JARVIS_BASE_PATH"):
    os.environ.pop(_name, None)
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="100000", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="100000",
                  JARVIS_MAX_TURNS_PER_CALL="100000")
import notify

# The real URL and secret from .env must never be used here.
SECRET = "stub-secret-123"
os.environ["N8N_JARVIS_SECRET"] = SECRET
os.environ.pop("FAMILY_NAMES", None)
notify.BACKOFF_S = [0.3]   # fast retries for the tests
agent.on_note_saved = notify.enqueue

results = []


def check(name, condition):
    results.append(bool(condition))
    print(("  ok   " if condition else "  FAIL ") + name)


# ---------- the stub webhook: answers from a script, records every request ----------
script, received = [], []


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        received.append({"secret": self.headers.get("X-Jarvis-Secret"), "body": json.loads(body)})
        status, reply, delay = script.pop(0) if script else (200, {"saved": True}, 0)
        time.sleep(delay)
        data = json.dumps(reply).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the client already gave up (the timeout test)

    def log_message(self, *args):
        pass


stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=stub.serve_forever, daemon=True).start()
STUB_URL = f"http://127.0.0.1:{stub.server_address[1]}/webhook/jarvis-notes"
os.environ["N8N_JARVIS_WEBHOOK_URL"] = STUB_URL


def outbox(note_id):
    conn = db.connect()
    row = conn.execute("SELECT * FROM outbox WHERE note_id = ?", (note_id,)).fetchone()
    conn.close()
    return row


def wait_for(condition, seconds=8):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def events(call_id, kind):
    return [r["detail"] for r in db.replay(call_id) if r["kind"] == kind]


def new_call(caller_id=""):
    agent.start_call(caller_id=caller_id, caller_lines=None, owner_answers=[], verbose=False)
    return agent.call_state["call_id"]


print("success: the exact JSON shape, the secret header, logged notify_sent")
call = new_call()
received.clear()
agent.add_reminder("Rahul", "high", "VPS renewal")
nid = f"jarvis-{call}-reminder-1"
check("delivered once, status sent", wait_for(lambda: outbox(nid) and outbox(nid)["status"] == "sent") and len(received) == 1)
body = received[0]["body"]
check("shape: exactly the agreed keys", set(body) == {"note_id", "type", "call_id", "caller_name", "is_family", "urgency", "text", "summary"})
check("values: note_id, type, call_id int, name, not family, urgency, text as saved",
      body["note_id"] == nid and body["type"] == "reminder" and body["call_id"] == call and body["caller_name"] == "Rahul"
      and body["is_family"] is False and body["urgency"] == "high" and body["text"] == "VPS renewal")
check("summary: one line from code", body["summary"] == "📞 Rahul called: VPS renewal, wants a callback (urgent)")
check("X-Jarvis-Secret header sent", received[0]["secret"] == SECRET)
check("notify_sent logged", events(call, "notify_sent") == [f"{nid} (attempt 1)"])
agent.take_message("Rahul called; the invoice is ready.", caller_name="Rahul")
agent.take_message("Also, the server bill went up this month.", caller_name="Rahul")
check("messages: own counter, urgency null", wait_for(lambda: len(received) == 3)
      and [r["body"]["note_id"] for r in received[1:]] == [f"jarvis-{call}-message-1", f"jarvis-{call}-message-2"]
      and received[1]["body"]["urgency"] is None and received[1]["body"]["text"] == "Rahul called; the invoice is ready."
      and received[1]["body"]["summary"] == "📝 Rahul left a message: Rahul called; the invoice is ready.")

print("500, then a retry with the SAME note_id")
call = new_call()
received.clear()
script[:] = [(500, {"error": "boom"}, 0), (200, {"saved": True}, 0)]
agent.add_reminder("Priya", "low", "invoice")
nid = f"jarvis-{call}-reminder-1"
check("retried and delivered", wait_for(lambda: outbox(nid)["status"] == "sent"))
check("both attempts used the same note_id", [r["body"]["note_id"] for r in received] == [nid, nid])
check("notify_failed logged with the reason, then notify_sent",
      events(call, "notify_failed") == [f"{nid}: 5xx (500), attempt 1, retry in 0.3s"] and events(call, "notify_sent") == [f"{nid} (attempt 2)"])

print("duplicate reply counts as delivered")
call = new_call()
received.clear()
script[:] = [(200, {"saved": False, "duplicate": True}, 0)]
agent.take_message("Neha called about the website.", caller_name="Neha")
nid = f"jarvis-{call}-message-1"
check("200 + duplicate: true -> sent, no retry", wait_for(lambda: outbox(nid)["status"] == "sent") and len(received) == 1)
call = new_call()
script[:] = [(200, [{"saved": True}], 0)]
agent.take_message("Second note.", caller_name="Neha")
check("an n8n list-wrapped reply also counts", wait_for(lambda: outbox(f"jarvis-{call}-message-1")["status"] == "sent"))

print("failure reasons: 401, 400, bad reply, timeout, unreachable (never the secret)")
for status, reply, reason in ((401, {"error": "bad secret"}, "401"), (400, {"error": "bad body"}, "400"),
                              (200, {"ok": True}, "bad reply"), (503, {}, "5xx (503)")):
    call = new_call()
    script[:] = [(status, reply, 0)]
    notify.BACKOFF_S = [60]  # no retry during this check
    agent.take_message(f"note for {reason}", caller_name="Amit")
    nid = f"jarvis-{call}-message-1"
    check(f"{status} -> '{reason}', still pending", wait_for(lambda: outbox(nid)["attempts"] == 1)
          and outbox(nid)["status"] == "pending" and outbox(nid)["last_error"] == reason
          and events(call, "notify_failed")[0].startswith(f"{nid}: {reason}, attempt 1"))
old_timeout, notify.TIMEOUT_S = notify.TIMEOUT_S, 0.5
call = new_call()
script[:] = [(200, {"saved": True}, 2)]
agent.take_message("slow one", caller_name="Amit")
nid = f"jarvis-{call}-message-1"
check("timeout -> 'timeout'", wait_for(lambda: outbox(nid)["last_error"] == "timeout"))
notify.TIMEOUT_S = old_timeout
os.environ["N8N_JARVIS_WEBHOOK_URL"] = "http://127.0.0.1:9/nothing-here"
call = new_call()
agent.take_message("nobody home", caller_name="Amit")
nid = f"jarvis-{call}-message-1"
check("closed port -> 'unreachable'", wait_for(lambda: outbox(nid)["last_error"] == "unreachable"))
os.environ["N8N_JARVIS_WEBHOOK_URL"] = STUB_URL
conn = db.connect()
logged = " ".join(r[0] or "" for r in conn.execute("SELECT detail FROM events")) + " ".join(
    (r[0] or "") + (r[1] or "") for r in conn.execute("SELECT payload, last_error FROM outbox"))
conn.close()
check("the secret is in no log line and no outbox row", SECRET not in logged)

print("server restart: pending notes are retried by start()")
notify.BACKOFF_S = [0.3]
script.clear()
received.clear()
conn = db.connect()
conn.execute("UPDATE outbox SET next_try = 0 WHERE status = 'pending'")
conn.commit()
conn.close()
notify._worker["stop"] = True
notify._wake.set()
wait_for(lambda: not notify._worker["thread"].is_alive())
check("worker stopped (like a server shutdown); notes still pending", len(notify.pending()) >= 6)
notify.start()
check("start() delivers everything left over", wait_for(lambda: not notify.pending(), 10) and len(received) >= 6)

print("slow webhook never delays the call")
call = new_call()
script[:] = [(200, {"saved": True}, 3)]
started = time.time()
result = agent.add_reminder("Karan", "medium", "logo design")
spent = time.time() - started
check(f"saving took {spent * 1000:.0f} ms while the webhook takes 3 s", spent < 0.5 and result.startswith("Reminder saved"))
check("...and the note still arrives", wait_for(lambda: outbox(f"jarvis-{call}-reminder-1")["status"] == "sent", 8))

print("URL not set: off, nothing errors, nothing queued")
os.environ["N8N_JARVIS_WEBHOOK_URL"] = ""
call = new_call()
received.clear()
result = agent.take_message("Sam called.", caller_name="Sam")
check("save works, no outbox row, no request, no error event", result.startswith("Message saved")
      and outbox(f"jarvis-{call}-message-1") is None and not received and not events(call, "notify_failed"))
check("enabled/enqueue/start all say off", notify.enabled() is False and notify.enqueue({"note_id": "x", "call_id": 1}) is False
      and notify.start() is False)
os.environ["N8N_JARVIS_WEBHOOK_URL"] = STUB_URL

print("family: mapped name, or 'Family member'; is_family true")
os.environ["FAMILY_NAMES"] = "1111:Papa,2222:Mummy"
received.clear()
call = new_call("1111")  # the scripted ring gets no answer: a family call that goes on
agent.add_reminder()
agent.take_message("Come home early.", caller_name="Beta")
check("1111 -> Papa, is_family, family reminder text 'wants a callback'", wait_for(lambda: len(received) == 2)
      and [(r["body"]["caller_name"], r["body"]["is_family"], r["body"]["type"]) for r in received]
      == [("Papa", True, "reminder"), ("Papa", True, "message")]
      and received[0]["body"]["text"] == "wants a callback" and received[0]["body"]["summary"] == "🏠 Family call – Papa wants a callback"
      and received[1]["body"]["summary"] == "🏠 Family call – Papa left a message: Come home early.")
received.clear()
call = new_call("3333")
agent.take_message("Bring milk.")
check("unmapped 3333 -> 'Family member'", wait_for(lambda: len(received) == 1) and received[0]["body"]["caller_name"] == "Family member")
os.environ.pop("FAMILY_NAMES")

print("WhatsApp formats: family vs normal (Jatin, 2026-11-26)")
check("family message", notify.summary_line("message", "Papa", "Come home early.", None, True)
      == "🏠 Family call – Papa left a message: Come home early.")
check("family callback, urgent", notify.summary_line("reminder", "Papa", "wants a callback", "high", True)
      == "🏠 Family call – Papa wants a callback (urgent)")
check("family callback with a reason", notify.summary_line("reminder", "Family member", "electricity bill", "low", True)
      == "🏠 Family call – Family member wants a callback: electricity bill")
check("normal callers keep the old format", notify.summary_line("reminder", "Rahul", "VPS", "high")
      == "📞 Rahul called: VPS, wants a callback (urgent)"
      and notify.summary_line("message", "Neha", "The site is live.", None) == "📝 Neha left a message: The site is live.")

print("only the real server sends")
import importlib
src = open("server.py", encoding="utf-8").read()
check("server wires notify only under __main__ (tests/evals import server without it)",
      src.index("agent.on_note_saved = notify.enqueue") > src.index('if __name__ == "__main__":'))
check("run_evals never sets the hook", "on_note_saved" not in open("run_evals.py", encoding="utf-8").read())
check(".env.example: both names, no values", "N8N_JARVIS_WEBHOOK_URL=\n" in open(".env.example", encoding="utf-8").read()
      and "N8N_JARVIS_SECRET=\n" in open(".env.example", encoding="utf-8").read())

stub.shutdown()
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
