# Stub test for Jarvis's failure paths: provider down, config/request errors, and a stuck agent loop.
# It fakes the Groq client, so it makes ZERO API calls and costs zero tokens.
# It writes to a throwaway database in your temp folder, never jarvis.db.
#
# Run it from the project folder:   python test_provider_down.py
import os
import sys
import json
import time
import io
import tempfile
import contextlib
from types import SimpleNamespace as N
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.DB_FILE = os.path.join(tempfile.gettempdir(), "jarvis_test.db")
if os.path.exists(db.DB_FILE):
    os.remove(db.DB_FILE)

import agent
# Public-run settings (access.py) pinned for the tests, whatever .env says: no lock, no demo mode, no base path,
# caps high enough for the many calls a suite starts. The access tests set their own values.
for _name in ("JARVIS_ACCESS_CODE", "JARVIS_DEMO_MODE", "JARVIS_BASE_PATH"):
    os.environ.pop(_name, None)
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="100000", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="100000",
                  JARVIS_MAX_TURNS_PER_CALL="100000")
# These suites test the judge-language path (option 3 off). The option-3 tests in test_streaming.py switch it
# on themselves; .env may switch it on for live calls and evals.
ENV_CODE_LANGUAGE = agent.CODE_LANGUAGE
agent.CODE_LANGUAGE = False
import run_evals
# Use the error classes of whichever provider agent.py loaded (groq or openai),
# so this test checks the real mapping for the provider in your .env.
from agent import APIConnectionError, APIStatusError, BadRequestError
print(f"Provider under test: {agent.PROVIDER} ({agent.MODEL} / {agent.JUDGE_MODEL})")

time.sleep = lambda s: None  # skip retry waits and the eval's 3s pauses

REQ = httpx.Request("POST", "https://api.groq.com/test")
calls = {"n": 0}


# ---------- fake model behaviours ----------

def outage(**kw):
    calls["n"] += 1
    raise APIConnectionError(request=REQ)


def http_error(status):
    """A fake model that always fails with this HTTP status code."""
    def fake(**kw):
        calls["n"] += 1
        response = httpx.Response(status, request=REQ)
        if status == 400:
            raise BadRequestError("bad request", response=response, body=None)
        raise APIStatusError(f"error {status}", response=response, body=None)
    return fake


def reply(content=None, tool_calls=None):
    msg = N(role="assistant", content=content, tool_calls=tool_calls)
    return N(choices=[N(message=msg)], usage=N(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None))


def judge(lang="english"):
    return reply(json.dumps({"language": lang, "caller_agreed_to_connect": False, "reason_known": True}))


def is_judge_reply(item):
    content = getattr(item.choices[0].message, "content", None) or ""
    return "caller_agreed_to_connect" in content


def take(queue, for_judge):
    """The judge now runs alongside the model, so calls arrive in either order. Judge requests
    (the ones asking for JSON) get the next judge reply in the script; model requests the next other one."""
    for i, item in enumerate(queue):
        if is_judge_reply(item) == for_judge:
            return queue.pop(i)
    return None


def scripted(responses, then=None):
    """Return scripted responses (judge replies to judge requests, the rest to model requests);
    once the right kind runs out, use `then` (default: an outage)."""
    queue = list(responses)
    def fake(**kw):
        calls["n"] += 1
        item = take(queue, "response_format" in kw)
        if item is not None:
            return item
        if then is not None:
            return then()
        raise APIConnectionError(request=REQ)
    return fake


def tool_call(name, args):
    return N(id=f"t{calls['n']}", type="function", function=N(name=name, arguments=json.dumps(args)))


# ---------- helpers ----------

def events(call_id):
    return [(r["kind"], r["detail"]) for r in db.replay(call_id)]


def said(call_id):
    return [d for k, d in events(call_id) if k == "jarvis_said"]


results = []
def check(name, condition):
    results.append((name, bool(condition)))
    print(("  ok   " if condition else "  FAIL ") + name)


def run(fake, lines, caller_id=""):
    agent.client.chat.completions.create = fake
    calls["n"] = 0
    return agent.run_call(caller_id=caller_id, caller_lines=list(lines), owner_answers=[], verbose=False)


real_save = db.save_reminder
def broken_save(*a, **k):
    raise RuntimeError("disk full")


# --- A: outage from the first call ---
print("A. outage on the first model call")
r = run(outage, ["hi this is priya", "bye"])
ev = events(r["call_id"])
check("no exception, outcome provider_unavailable", r["outcome"] == "provider_unavailable")
check("provider_down flag returned", r["provider_down"])
check("event provider_unavailable logged", any(k == "provider_unavailable" for k, _ in ev))
# The judge runs alongside the model now, so both retry side by side (5 tries each, same ~30 s wait
# for the caller); after that the flag stops every later call, including turn 2's.
check("5 tries each for judge and model (side by side), then no more calls (flag short-circuits)", calls["n"] == 10)
check("SAVED English line said (claims 'noted', and the note exists)", "I've noted that you called" in said(r["call_id"])[-1])
check("failed_call note saved, marked INCOMPLETE", any(x["kind"] == "failed_call" and x["content"].startswith("INCOMPLETE") for x in r["reminders"]))
check("loop stopped after first caller line", sum(k == "caller_said" for k, _ in ev) == 1)

# --- B: no-retry HTTP errors: 401/403/404 config, 413/422 request ---
print("B. no-retry HTTP errors")
for status, event in [(401, "config_error"), (403, "config_error"), (404, "config_error"),
                      (413, "request_error"), (422, "request_error")]:
    r = run(http_error(status), ["hello, jatin hai kya"])
    ev = events(r["call_id"])
    check(f"{status}: exactly 1 attempt, '{event}' event with status, note + outcome",
          calls["n"] == 1
          and any(k == event and d.startswith(str(status)) for k, d in ev)
          and not any(k == "provider_unavailable" for k, _ in ev)
          and r["outcome"] == "provider_unavailable" and r["reminders"])

# --- B2: 400 is NOT swallowed: it still reaches run_agent's own handling ---
print("B2. 400 bad request keeps its old path")
r = run(scripted([judge()], then=lambda: (_ for _ in ()).throw(
    BadRequestError("bad", response=httpx.Response(400, request=REQ), body=None))), ["hi this is priya"])
check("400 -> 'didn't catch that' fallback, call not ended as provider failure",
      said(r["call_id"])[-1].startswith("Sorry, didn't catch that") and not r["provider_down"])

# --- C: reminder saved earlier, THEN retries exhausted ---
print("C. reminder saved, then outage")
remind = tool_call("add_reminder", {"caller_name": "Rahul", "reason": "VPS renewal", "urgency": "low"})
# (No model reply after the save: since step 4 the follow-up is code-spoken, so the turn ends on the save.)
r = run(scripted([judge(), reply(tool_calls=[remind])]),
        ["hi this is rahul, save a reminder about the vps", "one more thing"])
kinds = [x["kind"] for x in r["reminders"]]
check("earlier reminder survives", "reminder" in kinds)
check("failed_call note added too", "failed_call" in kinds)
check("outcome overwritten to provider_unavailable", r["outcome"] == "provider_unavailable")
check("note carries the name and reason we had",
      any(x["kind"] == "failed_call" and x["caller_name"] == "Rahul" and "VPS renewal" in x["content"] for x in r["reminders"]))
# Step 4 made turn 1 end on the save (code-spoken follow-up), so prove the outage is still reached in turn 2.
ev = events(r["call_id"])
check("outage reached in turn 2: provider_unavailable logged after the follow-up",
      [k for k, _ in ev].count("provider_unavailable") >= 1
      and [d for k, d in ev if k == "fixed_line"][:1] == ["follow-up: reminder"]
      and [k for k, _ in ev].index("provider_unavailable") > [d for _, d in ev].index("follow-up: reminder"))
check("caller heard the follow-up, then the 'noted' fallback line",
      said(r["call_id"])[1:] == [agent.FOLLOW_UP_LINES["reminder"]["english"], agent.LINES_NOTE_SAVED["english"]])

# --- D: save_reminder itself throws -> the NOT-saved line, no false claim ---
print("D. note save fails")
db.save_reminder = broken_save
r = run(outage, ["hello"])
db.save_reminder = real_save
ev = events(r["call_id"])
line = said(r["call_id"])[-1]
check("no exception, a line is still said", line.startswith("Sorry, something's gone wrong"))
check("NOT-saved line used: no false 'noted' claim", "noted" not in line and "can't take your details" in line)
check("warn event explains the missing note", any(k == "warn" and "failed-call note" in d for k, d in ev))
check("no failed_call row exists", not r["reminders"])

print("D2. note save fails, Hinglish caller")
db.save_reminder = broken_save
r = run(scripted([judge("hinglish")]), ["mera naam Suresh hai"])
db.save_reminder = real_save
line = said(r["call_id"])[-1]
check("Hinglish NOT-saved line", line.startswith("Sorry, mere side kuch gadbad ho gayi hai, abhi main aapki details nahi"))
check("no 'note kar liya' claim", "note kar liya" not in line)

# --- E: Hinglish caller, note saved ---
print("E. hinglish line")
r = run(scripted([judge("hinglish")]), ["mera naam Suresh hai"])
check("Hinglish SAVED line", "note kar liya" in said(r["call_id"])[-1])

# --- G: agent loop stuck (MAX_STEPS tool calls, never a reply) ---
print("G. agent stuck at MAX_STEPS")
offer = lambda: reply(tool_calls=[tool_call("caller_wants_jatin", {})])
r = run(scripted([judge()] + [offer() for _ in range(agent.MAX_STEPS)]), ["can i talk to jatin", "hello?"])
ev = events(r["call_id"])
check("outcome agent_stuck, not provider_unavailable", r["outcome"] == "agent_stuck" and not r["provider_down"])
check("agent_stuck event logged", any(k == "agent_stuck" for k, _ in ev))
check("SAVED line said and recorded in the log", "I've noted that you called" in said(r["call_id"])[-1])
check("failed_call note saved", any(x["kind"] == "failed_call" for x in r["reminders"]))
check("call ended after that turn", sum(k == "caller_said" for k, _ in ev) == 1)

print("G2. agent stuck AND note save fails")
db.save_reminder = broken_save
r = run(scripted([judge()] + [offer() for _ in range(agent.MAX_STEPS)]), ["can i talk to jatin"])
db.save_reminder = real_save
line = said(r["call_id"])[-1]
check("NOT-saved line, no false 'noted' claim", "noted" not in line and "can't take your details" in line)

# --- F: run_evals stops cleanly with a partial score ---
print("F. run_evals during an outage")
agent.client.chat.completions.create = outage
sys.argv = ["run_evals.py"]
out = io.StringIO()
with contextlib.redirect_stdout(out):
    run_evals.main()
text = out.getvalue()
check("first scenario FAIL", [l for l in text.splitlines() if l.startswith(("PASS", "FAIL"))][0].startswith("FAIL"))
check("only one scenario attempted", sum(1 for l in text.splitlines() if l.startswith(("PASS", "FAIL"))) == 1)
check("partial score + stopped message", "Score: 0/1" in text and "Stopped early" in text)
check("failure reason 'provider unavailable'", "- provider unavailable" in text)

print("F2. run_evals marks a stuck scenario FAIL and keeps going")
stuck_scenario = {"name": "stub", "caller_id": "", "caller_lines": ["hi"], "owner_answers": [], "expect": {}}
agent.client.chat.completions.create = scripted([judge()] + [offer() for _ in range(agent.MAX_STEPS)])
_, failures = run_evals.run_one(stuck_scenario)
check("stuck counts as FAIL even with no other expectations", any("agent stuck" in f for f in failures))

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
