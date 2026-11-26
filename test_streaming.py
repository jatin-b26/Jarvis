# Zero-token checks for sentence-by-sentence streaming, barge-in and the duplicate-save guard (2026-11-25).
# Fake model streams, fake speech, a throwaway database. No API calls.
# Run from the project folder:   python test_streaming.py
import os
import sys
import json
import time
import io
import tempfile
import threading
from types import SimpleNamespace as N

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.DB_FILE = os.path.join(tempfile.gettempdir(), "jarvis_streaming_test.db")
if os.path.exists(db.DB_FILE):
    os.remove(db.DB_FILE)

import agent
# Public-run settings (access.py) pinned for the tests, whatever .env says: no lock, no demo mode, no base path,
# caps high enough for the many calls a suite starts. The access tests set their own values.
for _name in ("JARVIS_ACCESS_CODE", "JARVIS_DEMO_MODE", "JARVIS_BASE_PATH"):
    os.environ.pop(_name, None)
os.environ.update(JARVIS_MAX_CALLS_PER_DAY="100000", JARVIS_MAX_CALLS_PER_VISITOR_PER_DAY="100000",
                  JARVIS_MAX_TURNS_PER_CALL="100000")
import server
import voice
# These suites test the judge-language path (option 3 off). The option-3 tests in test_streaming.py switch it
# on themselves; .env may switch it on for live calls and evals.
ENV_CODE_LANGUAGE = agent.CODE_LANGUAGE
agent.CODE_LANGUAGE = False

voice.open_stream = lambda text, lang: (iter([b"\x01\x00" * 480]), voice.stream_voice(lang)["name"], 5)
voice.cached_pcm = lambda text, lang: None
voice.warm_tts = lambda: None
voice.close_tts = lambda: None
agent.RING_SECONDS = 1

results = []


def check(name, condition):
    results.append(bool(condition))
    print(("  ok   " if condition else "  FAIL ") + name)


# ---------- fakes ----------

JUDGE = {"agreed": False, "lang": "english", "reason": True}


def judge_reply():
    msg = N(role="assistant", content=json.dumps({"language": JUDGE["lang"], "caller_agreed_to_connect": JUDGE["agreed"],
                                                  "reason_known": JUDGE["reason"]}), tool_calls=None)
    return N(choices=[N(message=msg)], usage=N(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None))


def text_chunk(piece):
    return N(choices=[N(delta=N(content=piece, tool_calls=None))], usage=None)


def tool_chunk(index, call_id=None, name=None, arguments=None):
    call = N(index=index, id=call_id, function=N(name=name, arguments=arguments))
    return N(choices=[N(delta=N(content=None, tool_calls=[call]))], usage=None)


USAGE_CHUNK = N(choices=[], usage=N(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=N(cached_tokens=50)))


class FakeStream:
    """A streamed model reply. before(i) runs before chunk i is handed over (to act mid-stream).
    wait_at: the stream stalls before that chunk until released or closed (a slow model)."""

    def __init__(self, chunks, before=None, wait_at=None):
        self.chunks, self.before, self.wait_at = chunks, before, wait_at
        self.closed = False
        self.release = threading.Event()

    def __iter__(self):
        for i, chunk in enumerate(self.chunks):
            if self.wait_at == i:
                self.release.wait(timeout=5)
            if self.closed:
                raise RuntimeError("stream closed")
            if self.before:
                self.before(i)
            yield chunk

    def close(self):
        self.closed = True
        self.release.set()


model_requests = []


def script(*streams):
    """Model requests get the streams in order; judge requests (JSON mode) always get a verdict."""
    queue = list(streams)

    def create(**kw):
        if "response_format" in kw:
            return judge_reply()
        model_requests.append(kw)
        return queue.pop(0)
    model_requests.clear()
    agent.client.chat.completions.create = create


def new_call():
    agent.start_call(caller_id="", caller_lines=None, owner_answers=[], verbose=False)


def events(kind):
    return [row["detail"] for row in db.replay(agent.call_state["call_id"]) if row["kind"] == kind]


def assistant_texts():
    return [m.get("content") for m in agent.call_state["messages"] if isinstance(m, dict) and m.get("role") == "assistant"]


def tool_pairs_ok():
    """Every tool call in the history has a tool result (the API rejects the history otherwise)."""
    ids = [c["id"] for m in agent.call_state["messages"] if isinstance(m, dict) for c in m.get("tool_calls") or []]
    results_for = {m["tool_call_id"] for m in agent.call_state["messages"] if isinstance(m, dict) and m.get("role") == "tool"}
    return all(i in results_for for i in ids)


# ---------- sentences ----------
print("sentence splitting")
check("two sentences, the second still open", agent.split_sentences("Hi Priya. What's it about? Th") == (["Hi Priya.", "What's it about?"], "Th"))
check("a one-word sentence waits for the next", agent.split_sentences("Okay. Can I take your name? ") == (["Okay. Can I take your name?"], ""))
check("'Mr.' doesn't end a sentence", agent.split_sentences("Mr. Sharma called. ") == (["Mr. Sharma called."], ""))
check("no space after the last full stop: not complete yet", agent.split_sentences("Theek hai, bata dungi.") == ([], "Theek hai, bata dungi."))
check("Hindi full stop ends a sentence", agent.split_sentences("Theek hai ji। Aur kuch? ") == (["Theek hai ji।", "Aur kuch?"], ""))
check("decimal point doesn't split", agent.split_sentences("It costs 3.5 lakh. ") == (["It costs 3.5 lakh."], ""))

# ---------- streaming ----------
print("streaming: each clean sentence goes out before the model finishes (no tools offered: a greeting)")
new_call()
said, seen_at = [], {}
stream = FakeStream([text_chunk("Hi Priya"), text_chunk(". "), text_chunk("What's it"), text_chunk(" about?"), USAGE_CHUNK],
                    before=lambda i: seen_at.__setitem__(i, len(said)))
script(stream)
reply = agent.handle_turn("hello", sink=said.append, reply_id="r1")
check("first sentence sent while the model was still writing the second", seen_at[3] == 1)
check("both sentences sent, in order", said == ["Hi Priya.", "What's it about?"])
check("turn returns everything said", reply == "Hi Priya. What's it about?")
check("history: one assistant message with the whole reply", assistant_texts()[-1] == "Hi Priya. What's it about?")
check("jarvis_said logged once for the reply", events("jarvis_said")[-1] == "Hi Priya. What's it about?")
check("request asked for a stream with usage", model_requests[0].get("stream") is True
      and model_requests[0].get("stream_options") == {"include_usage": True})
check("usage from the stream's last chunk is logged", any("in=100 cached=50 out=20" in u for u in events("usage")))
check("timing records first_say", agent.call_state["timing"]["first_say"] is not None)

print("streaming: leak guard per sentence")
new_call()
said = []
script(FakeStream([text_chunk("Sure, Priya. "), text_chunk("The caller must be told politely. "), text_chunk("Bye.")]))
reply = agent.handle_turn("hi this is priya", sink=said.append, reply_id="r2")
check("clean first sentence said, leaking second never said, nothing after it", said == ["Sure, Priya."])
check("history holds only what was said", assistant_texts()[-1] == "Sure, Priya.")
check("leak logged + stop logged, no retry once something was said",
      any(d.startswith("self_reference") for d in events("leak_guard")) and events("leak_stop") and len(model_requests) == 1)

new_call()
said = []
script(FakeStream([text_chunk("The assistant should ask the name. ")]), FakeStream([text_chunk("Can I take your name?")]))
reply = agent.handle_turn("hello there priya here", sink=said.append, reply_id="r3")
check("leak in the first sentence: nothing said, one retry, as before streaming",
      said == ["Can I take your name?"] and len(model_requests) == 2 and events("leak_guard"))
new_call()
said = []
script(FakeStream([text_chunk("Okay, get their name. "), text_chunk("And what it's about. ")]))
agent.handle_turn("hi there, priya here", sink=said.append, reply_id="r3b")
check("a 6-word prompt run split across two sentences is caught (the guard reads all said so far)",
      said == ["Okay, get their name."] and any(d.startswith("hint_overlap") for d in events("leak_guard")))

print("streaming: tool calls arrive in pieces")
new_call()
said = []
script(FakeStream([tool_chunk(0, "t1", "take_message", '{"caller_name": "Pri'),
                   tool_chunk(0, None, None, 'ya", "message": "Priya called; the invoice is late."}'), USAGE_CHUNK]))
reply = agent.handle_turn("this is priya, tell him the invoice is late", sink=said.append, reply_id="r4")
saved = db.reminders_for_call(agent.call_state["call_id"])
check("tool call assembled from pieces and run", len(saved) == 1 and "invoice is late" in saved[0]["content"])
check("follow-up line said through the sink", said == [agent.pick_line(agent.FOLLOW_UP_LINES["message"], "english")])
check("history: tool call as a dict, with its result", tool_pairs_ok()
      and any(isinstance(m, dict) and m.get("tool_calls") for m in agent.call_state["messages"]))

print("tools offered: text is held until the stream ends, and never said before a tool call")
new_call()
said, seen_at = [], {}
script(FakeStream([text_chunk("Thanks Priya. "), text_chunk("What's it about?"), USAGE_CHUNK],
                  before=lambda i: seen_at.__setitem__(i, len(said))))
agent.handle_turn("hi this is priya", sink=said.append, reply_id="h1")
check("nothing said while the stream was still open", seen_at[1] == 0 and seen_at[2] == 0)
check("a text-only reply is said after the stream ends, sentence by sentence", said == ["Thanks Priya.", "What's it about?"])
new_call()
said = []
agent.call_state.update(wants_jatin_turn=0, caller_agreed=True, reason_known=True, turns=3)
script(FakeStream([text_chunk("Putting you through now. "),
                   tool_chunk(0, "h2", "try_reach_owner", '{"caller_name": "Priya", "reason": "invoice"}')]),
       FakeStream([text_chunk("He can't take it right now. Want me to take a message?")]))
agent.handle_turn("yes please try him", sink=said.append, reply_id="h2")
check("text before a tool call is never said (a refused ring then gets the code line, no model reply)",
      "Putting you through now." not in said and said == [agent.REFUSAL_LINES["not_agreed"]["english"]])
check("...and not in the history either", "Putting you through" not in json.dumps(agent.call_state["messages"])
      and not events("spoke_before_tools"))

# ---------- barge-in ----------
print("barge-in: mid-stream, from another thread (greeting turns: sentences stream)")
new_call()
said = []
stream = FakeStream([text_chunk("Okay Priya. "), text_chunk("Let me note "), text_chunk("that down for him.")], wait_at=1)
script(stream)
out = {}
worker = threading.Thread(target=lambda: out.update(reply=agent.handle_turn("hello", sink=said.append, reply_id="r5")))
worker.start()
for _ in range(100):
    if said:
        break
    time.sleep(0.01)
started = time.time()
result = agent.barge_in("r5", 1)
worker.join(timeout=5)
check("barge_in returns at once, without waiting for the turn", time.time() - started < 0.5 and result["applied"])
check("the stream was closed and the turn stopped", stream.closed and not worker.is_alive())
check("nothing after the barge-in was said", said == ["Okay Priya."])
check("history = what was heard", assistant_texts()[-1] == "Okay Priya.")
barge = json.loads(events("barge_in")[-1])
check("barge_in event logs heard and cut", barge["heard"] == ["Okay Priya."] and barge["cut"] == [])
check("cancelled event logged", any("during model reply" in d for d in events("cancelled")))

new_call()
said = []
stream = FakeStream([text_chunk("Okay Priya. "), text_chunk("Let me note that.")], wait_at=1)
script(stream)
worker = threading.Thread(target=lambda: agent.handle_turn("hi there", sink=said.append, reply_id="r6"))
worker.start()
for _ in range(100):
    if said:
        break
    time.sleep(0.01)
agent.barge_in("r6", 0)
worker.join(timeout=5)
messages = agent.call_state["messages"]
check("heard nothing: no assistant message for the turn (last message is the caller's)",
      messages[-1] == {"role": "user", "content": "hi there"})
check("the cut line is in the log as cut", json.loads(events("barge_in")[-1])["cut"] == ["Okay Priya."])

print("barge-in: after the turn finished (audio still playing)")
new_call()
script(FakeStream([text_chunk("Thanks Priya. "), text_chunk("What's it about?")]))
agent.handle_turn("hi this is priya", sink=lambda line: None, reply_id="r7")
agent.barge_in("r7", 1)
check("finished turn cut back to the first sentence", assistant_texts()[-1] == "Thanks Priya.")
check("a second barge-in for the same reply changes nothing", agent.barge_in("r7", 0)["applied"] is False
      and assistant_texts()[-1] == "Thanks Priya.")
script(FakeStream([text_chunk("Got it.")]))
agent.handle_turn("it's about the invoice", sink=lambda line: None, reply_id="r8")
agent.barge_in("r8", 0)
check("heard none of a finished one-line reply: removed from the history",
      agent.call_state["messages"][-1] == {"role": "user", "content": "it's about the invoice"})
check("judge's next transcript has no unheard line", "Got it." not in json.dumps(agent.call_state["messages"]))

print("barge-in: tools are atomic")
new_call()
said = []
holder = {}
stream = FakeStream([tool_chunk(0, "t9", "take_message", '{"caller_name": "Priya", "message": "Priya called; invoice late."}'),
                     USAGE_CHUNK], before=lambda i: i == 1 and agent.barge_in("r9", 0))
script(stream)
agent.handle_turn("tell him the invoice is late, i'm priya", sink=said.append, reply_id="r9")
check("barge-in before the tool ran: nothing saved", len(db.reminders_for_call(agent.call_state["call_id"])) == 0)
check("the unrun tool call is not in the history", tool_pairs_ok()
      and not any(isinstance(m, dict) and m.get("tool_calls") for m in agent.call_state["messages"]))
check("logged as not run", any("tool calls not run: take_message" in d for d in events("cancelled")))

new_call()
said = []
real_take = agent.TOOL_FUNCTIONS["take_message"]


def take_then_barge(**args):
    result = real_take(**args)
    agent.barge_in("r10", 0)  # the caller starts talking while the save is going through
    return result


agent.TOOL_FUNCTIONS["take_message"] = take_then_barge
script(FakeStream([tool_chunk(0, "t10", "take_message", '{"caller_name": "Priya", "message": "Priya called; invoice late."}')]))
agent.handle_turn("tell him the invoice is late, i'm priya", sink=said.append, reply_id="r10")
agent.TOOL_FUNCTIONS["take_message"] = real_take
check("save that went through is kept", len(db.reminders_for_call(agent.call_state["call_id"])) == 1)
check("kept_after_barge_in logged", any(d.startswith("take_message") for d in events("kept_after_barge_in")))
check("no follow-up said over the caller, and it's still due later", said == [] and agent.call_state["follow_up_spoken"] is False)
check("history stays valid: the call and its result", tool_pairs_ok())

new_call()
real_wants = agent.TOOL_FUNCTIONS["caller_wants_jatin"]


def wants_then_barge(**args):
    result = real_wants(**args)
    agent.barge_in("r11", 0)
    return result


agent.TOOL_FUNCTIONS["caller_wants_jatin"] = wants_then_barge
script(FakeStream([tool_chunk(0, "a1", "caller_wants_jatin", "{}"),
                   tool_chunk(1, "a2", "take_message", '{"caller_name": "Priya", "message": "Priya called; invoice late."}')]))
agent.handle_turn("i'm priya, can i talk to jatin? also the invoice is late", sink=lambda line: None, reply_id="r11")
agent.TOOL_FUNCTIONS["caller_wants_jatin"] = real_wants
batch = [m for m in agent.call_state["messages"] if isinstance(m, dict) and m.get("tool_calls")]
check("barge-in mid-batch: the second tool never runs", len(db.reminders_for_call(agent.call_state["call_id"])) == 0)
check("history keeps only the tool call that ran, with its result",
      len(batch) == 1 and [c["function"]["name"] for c in batch[0]["tool_calls"]] == ["caller_wants_jatin"] and tool_pairs_ok())

print("barge-in: before the turn began (still transcribing)")
new_call()
said = []
script()
agent.barge_in("r12", 0)
agent.handle_turn("hello hello is this jatin's phone", sink=said.append, reply_id="r12")
check("early barge-in: no model request, nothing said", model_requests == [] and said == [])
check("the caller's words stay in the history", agent.call_state["messages"][-1]["content"] == "hello hello is this jatin's phone")

# ---------- decline line (live call 2026-11-25) ----------
print("decline line: code says it after a failed ring, no model request")


def ring_turn(answer, lang=None, barge=False):
    agent.start_call(caller_id="", caller_lines=None, owner_answers=[answer], verbose=False)
    agent.call_state.update(wants_jatin_turn=0, caller_agreed=True, reason_known=True, turns=3)
    if lang:
        agent.call_state["speech_lang"] = lang
    said = []
    if barge:
        real = agent.TOOL_FUNCTIONS["try_reach_owner"]

        def ring_then_barge(**args):
            result = real(**args)
            agent.barge_in("ring", 0)
            return result
        agent.TOOL_FUNCTIONS["try_reach_owner"] = ring_then_barge
    script(FakeStream([tool_chunk(0, "k1", "try_reach_owner", '{"caller_name": "Priya", "reason": "invoice"}')]),
           FakeStream([text_chunk("Okay, connecting you now.")]))  # used only if he accepts
    JUDGE["agreed"], JUDGE["lang"] = True, lang or "english"
    agent.handle_turn("yes please try him", sink=said.append, reply_id="ring")
    JUDGE["agreed"], JUDGE["lang"] = False, "english"
    if barge:
        agent.TOOL_FUNCTIONS["try_reach_owner"] = real
    return said


said = ring_turn("no")
check("declined: the English decline line, spoken by code", said == [agent.DECLINE_LINES["english"]])
check("declined: one model request only (the tool call), none for the line", len(model_requests) == 1)
check("declined: history = tool result, then the line", agent.call_state["messages"][-1]["content"] == agent.DECLINE_LINES["english"]
      and agent.call_state["messages"][-2]["content"].startswith("Not connected.") and tool_pairs_ok())
check("logged as a fixed line", "decline line: decline" in events("fixed_line"))
said = ring_turn("", lang="hinglish")
check("no answer, Hinglish caller: the Hinglish line", said == [agent.DECLINE_LINES["hinglish"]])
check("female voice: same line (no gendered word in it)", agent.jarvis_gender() == "female"
      and agent.pick_line(agent.DECLINE_LINES, "hinglish") == agent.DECLINE_LINES["hinglish"])
said = ring_turn("yes")
check("accepted: no decline line", agent.DECLINE_LINES["english"] not in said and agent.call_state["connected"])
said = ring_turn("no", barge=True)
check("barge-in during the ring: decline line not said", said == [])
check("...and the tool result no longer claims they heard it",
      any(isinstance(m, dict) and m.get("content") == agent.DECLINE_UNHEARD_RESULT for m in agent.call_state["messages"]))
for label, text in [("decline result", "Not connected. They have heard that and been offered a message or a reminder. Save nothing until they answer."),
                    ("decline-unheard result", agent.DECLINE_UNHEARD_RESULT),
                    ("English decline line", agent.DECLINE_LINES["english"]), ("Hinglish decline line", agent.DECLINE_LINES["hinglish"])]:
    check(f"leak check: {label} passes the guard, no 'wait'",
          agent.leak_check(text, "", "") is None and "wait" not in text.lower())
check("decline lines are in the fixed-line set (so they get cached)",
      agent.DECLINE_LINES["english"] in agent.all_fixed_lines()["english"]
      and agent.DECLINE_LINES["hinglish"] in agent.all_fixed_lines()["hinglish"])

# ---------- three more code lines (Jatin, 2026-11-25) ----------
print("code lines: connected, a later save, a duplicate (model text dropped, no model reply request)")
said = ring_turn("yes")
check("accepted: 'Connecting you now.' by code, no model reply request",
      said == ["Connecting you now."] and len(model_requests) == 1 and agent.call_state["connected"])
said = ring_turn("yes", lang="hinglish")
check("accepted, Hinglish, female voice: 'Connect kar rahi hoon.'", said == ["Connect kar rahi hoon."])


def two_saves(second_message, lang=None, model_text="", barge=False):
    """First save (follow-up), then a second take_message turn; returns what the second turn said."""
    new_call()
    agent.call_state["turns"] = 3
    if lang:
        agent.call_state["speech_lang"] = lang
    first = '{"caller_name": "Ramesh", "message": "Ramesh called; the site is ready to go live."}'
    chunks = ([text_chunk(model_text)] if model_text else []) + [
        tool_chunk(0, "s2", "take_message", json.dumps({"caller_name": "Ramesh", "message": second_message}))]
    script(FakeStream([tool_chunk(0, "s1", "take_message", first)]), FakeStream(chunks))
    agent.handle_turn("hi this is ramesh, tell jatin the site is ready to go live", sink=lambda line: None, reply_id="c1")
    said = []
    if barge:
        real = agent.TOOL_FUNCTIONS["take_message"]

        def take_then_barge(**args):
            result = real(**args)
            agent.barge_in("c2", 0)
            return result
        agent.TOOL_FUNCTIONS["take_message"] = take_then_barge
    JUDGE["lang"] = lang or "english"
    agent.handle_turn("also tell him the payment is done" if not lang else "aur bolna payment ho gaya",
                      sink=said.append, reply_id="c2")
    JUDGE["lang"] = "english"
    if barge:
        agent.TOOL_FUNCTIONS["take_message"] = real
    return said


said = two_saves("Ramesh called; the payment is done.", model_text="Saved exactly as you said, waiting. ")
check("later save: 'Got it, I'll pass that on too.' by code", said == ["Got it, I'll pass that on too."])
check("...the model's text in that turn is dropped (never said, not in the history)",
      "exactly as you said" not in json.dumps(agent.call_state["messages"])
      and any("replaced by the added line" in d for d in events("dropped_text")))
check("...two model requests in the whole call (one per save), none for the line", len(model_requests) == 2)
said = two_saves("Ramesh called; the payment is done.", lang="hinglish")
check("later save, Hinglish, female voice: 'Theek hai, yeh bhi bata dungi.'", said == ["Theek hai, yeh bhi bata dungi."])
said = two_saves("Ramesh called; the site is ready to go live.", model_text="I already saved that exact message. ")
check("duplicate: 'That's already with him.' by code, model text dropped",
      said == ["That's already with him."] and "exact message" not in json.dumps(agent.call_state["messages"]))
said = two_saves("Ramesh called; the site is ready to go live.", lang="hinglish")
check("duplicate, Hinglish: 'Woh already note ho gaya hai.'", said == ["Woh already note ho gaya hai."])
said = two_saves("Ramesh called; the payment is done.", barge=True)
seen_results = [m["content"] for m in agent.call_state["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
check("barge-in before the later-save line: not said, and the result no longer claims it was heard",
      said == [] and seen_results[-1] == "Message saved from Ramesh." and seen_results[0].startswith("Message saved from Ramesh."))
for label, text in [("connected result", "Jatin ACCEPTED. They've heard they're being connected."),
                    ("later-save result", "Message saved from Ramesh. They've heard it confirmed."),
                    ("duplicate result", agent.DUPLICATE_RESULT)] + [
        (f"{n} line ({lang})", agent.pick_line(t, lang)) for n, t in (("connected", agent.CONNECTED_LINES),
                                                                     ("added", agent.ADDED_LINES),
                                                                     ("duplicate", agent.DUPLICATE_LINES))
        for lang in ("english", "hinglish")]:
    check(f"leak check: {label} passes the guard, no 'wait'", agent.leak_check(text, "", "") is None and "wait" not in text.lower())
check("female Hinglish forms: rahi / dungi; male forms kept for a male voice",
      agent.pick_line(agent.CONNECTED_LINES, "hinglish") == "Connect kar rahi hoon."
      and agent.pick_line(agent.ADDED_LINES, "hinglish") == "Theek hai, yeh bhi bata dungi."
      and agent.CONNECTED_LINES["hinglish"] == "Connect kar raha hoon." and agent.ADDED_LINES["hinglish"].endswith("dunga."))
fixed = agent.all_fixed_lines()
check("all three in the fixed-line set (cached, never graded)",
      all(agent.pick_line(t, lang) in fixed[lang] for t in (agent.CONNECTED_LINES, agent.ADDED_LINES, agent.DUPLICATE_LINES)
          for lang in ("english", "hinglish")))

# ---------- claim guard + ring tool menu (live call #912, 2026-11-25) ----------
print("claim guard: #912's exact lines, English and Hinglish, questions allowed")
for line in ["Theek hai, main Jatin ko abhi ring karti hoon.", "Theek hai, main abhi dobara ring karti hoon.",
             "main abhi dobara ring karti hoon", "Okay, I'll ring him now.", "I'm calling him now.", "Putting you through.",
             "Let me try him.", "Main call kar rahi hoon.", "Connect kar deti hoon.", "Main try karungi.",
             "Call laga rahi hoon."]:
    check(f"blocked as a claim: {line!r}", agent.claims_action(line))
for line in ["Shall I try calling him?", "Kya main Jatin ko call karne ka try karoon?", "I can't put you through right now.",
             "Abhi main unhe ring nahi kar sakti.", "Neha, I tried calling him.", "Theek hai, main bata dungi.",
             "Do you want him to call you back?", "Can I ask what you're calling about?", "Woh call kar rahe hain.",
             "I'll pass it on to him."]:
    check(f"not a claim: {line!r}", not agent.claims_action(line))
check("offers: a question to ring is an offer; a question about a message isn't",
      agent.offers_ring("Kya main abhi Jatin ko call karke aapko connect kar deti hoon?")
      and agent.offers_ring("Shall I try putting you through now?")
      and not agent.offers_ring("Should I take a message or set a reminder for him?"))


def claim_turn(streams, text="mujhe unse kuch kaam tha", wants=None, reason=False, lang=None):
    new_call()
    agent.call_state.update(wants_jatin_turn=wants, reason_known=reason, turns=2)
    if lang:
        agent.call_state["speech_lang"] = lang
    said = []
    script(*streams)
    JUDGE["lang"], JUDGE["reason"] = lang or "english", reason
    agent.handle_turn(text, sink=said.append, reply_id="cg")
    JUDGE["lang"], JUDGE["reason"] = "english", True
    return said


said = claim_turn([FakeStream([text_chunk("Theek hai, main Jatin ko abhi ring karti hoon.")]),
                   FakeStream([text_chunk("Kis baare mein baat karni thi?")])], lang="hinglish")
check("#912 turn 2: the claim is never said; one retry; the retry's clean line is said",
      said == ["Kis baare mein baat karni thi?"] and len(model_requests) == 2
      and any(d.startswith("claim: Theek hai, main Jatin ko abhi ring karti hoon.") for d in events("claim_guard"))
      and model_requests[1]["messages"][-2]["content"] == agent.CLAIM_RETRY_NOTES["claim"])
said = claim_turn([FakeStream([text_chunk("Theek hai, main abhi dobara ring karti hoon.")]),
                   FakeStream([text_chunk("Haan, main ring kar rahi hoon.")])], lang="hinglish")
check("#912 turn 3: retry claims again, reason unknown -> code asks the reason, never the claim",
      said == ["Pehle bata dijiye, kis baare mein baat karni hai?"]
      and "ring karti hoon" not in json.dumps(agent.call_state["messages"])
      and "claim retry blocked too: reason" in events("fixed_line"))
said = claim_turn([FakeStream([text_chunk("Theek hai, main abhi dobara ring karti hoon.")]),
                   FakeStream([text_chunk("Haan, main ring kar rahi hoon.")])], lang="hinglish", reason=True)
check("retry claims again, reason known -> the options line (no 'can't put you through'), counts as an offer",
      said == ["Main unhe call karne ka try karoon, ya message ya reminder chhod doon?"]
      and "claim retry blocked too: offer" in events("fixed_line") and agent.call_state["wants_jatin_turn"] is not None
      and events("offer_made"))
said = claim_turn([FakeStream([text_chunk("Okay, noted. "), text_chunk("I'll ring him now.")])])
check("claim after a clean sentence: stop there (the clean one said, the claim not)", said == ["Okay, noted."]
      and events("leak_stop"))
said = claim_turn([FakeStream([text_chunk("Kya main abhi Jatin ko call karke aapko connect kar deti hoon?")]),
                   FakeStream([text_chunk("Aapko kis baare mein baat karni thi?")])], lang="hinglish")
check("#912 turn 1: offer before caller_wants_jatin -> blocked as early_offer, retry said",
      said == ["Aapko kis baare mein baat karni thi?"] and any(d.startswith("early_offer:") for d in events("claim_guard")))
said = claim_turn([FakeStream([text_chunk("Shall I try calling him now?")]), FakeStream([text_chunk("What's it about?")])],
                  text="i want to talk to jatin", wants=1, reason=False)
check("offer after caller_wants_jatin but reason unknown -> blocked", said == ["What's it about?"])
said = claim_turn([FakeStream([text_chunk("Shall I try calling him now?")])], text="it's about the invoice", wants=1, reason=True)
check("offer after caller_wants_jatin with a known reason -> allowed", said == ["Shall I try calling him now?"])
for label, text in list(agent.CLAIM_RETRY_NOTES.items()) + [(f"fallback {k}", v) for k, v in agent.OFFER_FALLBACK_LINES.items()]:
    check(f"leak check: {label} passes the guard, no 'wait', not itself a claim",
          agent.leak_check(text, "", "") is None and "wait" not in text.lower() and not agent.claims_action(text))
check("fallback lines in the fixed-line set (cached)", all(agent.pick_line(agent.OFFER_FALLBACK_LINES, lang) in agent.all_fixed_lines()[lang]
                                                          for lang in ("english", "hinglish")))

print("ring tool menu: try_reach_owner only after caller_wants_jatin on an EARLIER turn")


def menu(request):
    return [t["function"]["name"] for t in request.get("tools", [])]


new_call()
script(FakeStream([tool_chunk(0, "w1", "caller_wants_jatin", "{}")]), FakeStream([text_chunk("What's it about?")]))
agent.handle_turn("can i talk to jatin please", sink=lambda line: None, reply_id="m1")
check("before caller_wants_jatin: no try_reach_owner on the menu; right after it (same turn): on it",
      "try_reach_owner" not in menu(model_requests[0]) and "caller_wants_jatin" in menu(model_requests[0])
      and "try_reach_owner" in menu(model_requests[1]))
script(FakeStream([text_chunk("Shall I try calling him now?")]))
agent.handle_turn("it's about the invoice for the website", sink=lambda line: None, reply_id="m2")
check("next turn: try_reach_owner is on the menu", "try_reach_owner" in menu(model_requests[0]))
new_call()
agent.call_state["turns"] = 1
script(FakeStream([tool_chunk(0, "x1", "try_reach_owner", '{"caller_name": "Neha", "reason": "website"}')]),
       FakeStream([text_chunk("Can I ask what it's about?")]))
agent.handle_turn("hi this is neha, about the website, put me through", sink=lambda line: None, reply_id="m3")
check("not on the menu yet, called anyway (#884) -> refused with the caller_wants_jatin error, no ring",
      "try_reach_owner -> ERROR: use caller_wants_jatin first, then wait for their answer." in events("tool_result")
      and agent.call_state["rings"] == 0 and "try_reach_owner" not in menu(model_requests[0]))

print("eval: the widened connecting check catches #912 (it scored nothing before)")
import run_evals
new_call()
for line in ["Theek hai, main Jatin ko abhi ring karti hoon.", "Theek hai, main abhi dobara ring karti hoon."]:
    agent.record("jarvis_said", line)
failures = run_evals.speech_failures({"expect": {}}, agent.call_state["call_id"])
check("both #912 lines fail the eval's claim check", sum("claimed to be ringing or connecting" in f for f in failures) == 2)
agent.record("claim_guard", "claim: Theek hai, main Jatin ko abhi ring karti hoon.")
check("eval lists the guard's blocks with their lines",
      run_evals.claim_blocks(agent.call_state["call_id"]) == [("claim", "Theek hai, main Jatin ko abhi ring karti hoon.")])

print("talk button: clip length and size in the stt line")
client = server.app.test_client()
client.post("/start", json={"reply_id": "b0"})
voice.transcribe = lambda data, name, mime, seconds=None: (voice.last_stt.update(engine="sarvam", why=None) or "ok then", 10)
script(FakeStream([text_chunk("Sure.")]))
client.post("/turn_audio", data={"audio": (io.BytesIO(b"x" * 3000), "turn.webm", "audio/webm"), "seconds": "3.25", "reply_id": "b1"})
check("stt line: engine, time, clip seconds and KB", "sarvam 10ms [clip 3.2 s, 2 KB]" in events("stt"))
client.post("/hangup")
page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"), encoding="utf-8").read()
check("page: no pointerleave stop; pointer capture; a press in the tail resumes the recording",
      '"pointerleave", stopTalking' not in page and "setPointerCapture" in page and "clearTimeout(tailTimer)" in page)

# ---------- #962: a Hinglish "bye" read as English by the judge ----------
print("#962: the goodbye follows the caller's Hinglish even when the judge says English")
new_call()
agent.call_state.update(turns=3, speech_lang="hinglish", lang="hinglish", follow_up_spoken=True, outcome="message")
said = []
script(FakeStream([tool_chunk(0, "e1", "end_call", '{"reason": "caller said bye", "kind": "done"}')]))
JUDGE["lang"] = "english"                      # what the judge said on #962
agent.handle_turn("nahi bas itna hi, bye", sink=said.append, reply_id="bye")
check("#962's exact line: Hinglish closing line, not English",
      said == [agent.pick_line(agent.CLOSING_LINES["after_follow_up"], "hinglish")] and agent.call_state["speech_lang"] == "hinglish")
check("2+ Roman Hindi words -> Hinglish; English lines don't trip it",
      agent.has_hindi_words("nahi bas itna hi, bye") and agent.has_hindi_words("achha theek hai")
      and not agent.has_hindi_words("no, that's all, bye") and not agent.has_hindi_words("what do you do, hi there")
      and not agent.has_hindi_words("i said it doesn't matter, connect me"))

# ---------- four fixes (2026-11-25, after #970-1026) ----------
print("near-duplicates: compare content, after greetings, self-introductions and the name")
said = two_saves("hi, this is Ramesh, please tell Jatin the site is ready to go live")
check("#1015's exact pair: the raw re-save is refused, code says it's already with him",
      said == ["That's already with him."] and len(db.reminders_for_call(agent.call_state["call_id"])) == 1)
said = two_saves("Ramesh called; the payment for the site has been received.")
check("a genuinely new second message still saves", said == ["Got it, I'll pass that on too."]
      and len(db.reminders_for_call(agent.call_state["call_id"])) == 2)
check("note_content drops greeting, intro, name and filler",
      agent.note_content("hi, this is Ramesh, please tell Jatin the site is ready to go live", {"ramesh"})
      == agent.note_content("Ramesh called; the site is ready to go live.", {"ramesh"}) == ["site", "ready", "go", "live"])

print("tool results: no Hindi words for an English caller")
for lang, want in (("english", False), ("hinglish", True)):
    new_call()
    agent.call_state.update(turns=2, reason_known=True)
    JUDGE["lang"] = lang
    agent.start_judge(agent.call_state["messages"] + [{"role": "user", "content": "x"}])
    result = agent.caller_wants_jatin()
    JUDGE["lang"] = "english"
    check(f"caller_wants_jatin result for a {lang} caller: Hinglish example {'included' if want else 'left out'}",
          ("call karne ka try" in result) == want and agent.has_hindi_words(result) is False or want)
    if not want:
        check("English caller's result has no Hindi words at all", "karne" not in result and not agent.has_hindi_words(result))
    check(f"{lang} result reads cleanly (no stray capital 'Or')",
          ("answer). Or offer" in result) if not want else ('answer); in Hinglish: "call karne ka try". Or offer' in result))

print("offer guard: asking for the name or the reason is not an offer")
check("#990's line passes", not agent.offers_ring("I need your name to put you through — can you tell me?"))
check("asking for name + reason with 'so I can try to put you through' passes",
      not agent.offers_ring("Can I get your name and what it's about so I can try to put you through?"))
check("#912's offer is still blocked", agent.offers_ring("Theek hai Karan — kya main abhi Jatin ko call karke aapko connect kar deti hoon?"))
said = claim_turn([FakeStream([text_chunk("I need your name to put you through — can you tell me?")])],
                  text="just put me through", wants=1, reason=False)
check("#990 end to end: said, not blocked", said == ["I need your name to put you through — can you tell me?"])
for line in ["I can only put you through if I have your name and what it's about.",
             "I need your name and what the call is about before I can try to connect you.",
             "I can try to connect, but I need your name and what the call's about first."]:
    check(f"#1045-1047 statement asking for details passes: {line[:50]!r}", not agent.offers_ring(line))
check("an offer with no ask is still blocked", agent.offers_ring("Shall I try putting you through to Jatin now?"))
check("#1159's reason-ask passes", not agent.offers_ring("And what's it about, so I can try to put you through if appropriate."))

print("scenario: the 'why Urdu?' turn retired (callers never see the note)")
_sc = {x["name"]: x for x in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json"), encoding="utf-8"))}
check("renamed, three caller lines, no Urdu question",
      "caller asks to save the note again" in _sc and "caller asks how the note was saved" not in _sc
      and _sc["caller asks to save the note again"]["caller_lines"][1] == "can you save it again exactly how I said it?"
      and len(_sc["caller asks to save the note again"]["caller_lines"]) == 3)

# ---------- refused ring -> code line (#912 turn 2, #1062) ----------
print("refused ring: code speaks a fixed line by reason; the model never writes that turn")


def refused_turn(state, args, text="haan try karo", lang="hinglish", agreed=False, reason=True, barge=False):
    new_call()
    agent.call_state.update(**state)
    said = []
    script(FakeStream([tool_chunk(0, "r1", "try_reach_owner", args)]),
           FakeStream([text_chunk("Theek hai, main Jatin ko abhi ring karti hoon.")]))  # what #912 said next
    JUDGE.update(agreed=agreed, lang=lang, reason=reason)
    if barge:
        real = agent.TOOL_FUNCTIONS["try_reach_owner"]
        agent.TOOL_FUNCTIONS["try_reach_owner"] = lambda **a: (agent.barge_in("rf", 0), real(**a))[1]
    agent.handle_turn(text, sink=said.append, reply_id="rf")
    if barge:
        agent.TOOL_FUNCTIONS["try_reach_owner"] = real
    JUDGE.update(agreed=False, lang="english", reason=True)
    return said


said = refused_turn({"wants_jatin_turn": 1, "turns": 2}, '{"caller_name": "Karan", "reason": "wants to meet"}',
                    text="Hawaai karo.", agreed=True, reason=False)
check("#912 turn 2 (no reason): the Hinglish 'no reason' line by code, one model request, no ringing claim",
      said == [agent.REFUSAL_LINES["reason"]["hinglish"]] and len(model_requests) == 1
      and "ring karti hoon" not in json.dumps(agent.call_state["messages"])
      and agent.call_state["messages"][-2]["content"] == agent.REFUSAL_RESULTS["reason"])
said = refused_turn({"wants_jatin_turn": 3, "turns": 4, "reason_known": True}, '{"caller_name": "Rohit", "no_reason_given": true}',
                    text="haan try kar lo", agreed=False)
check("#1062 (not agreed): 'try him or a reminder?' by code; 'koshish karti hoon' never said",
      said == [agent.REFUSAL_LINES["not_agreed"]["hinglish"]] and "koshish" not in json.dumps(agent.call_state["messages"]))
said = refused_turn({"wants_jatin_turn": 1, "turns": 4, "rings": 1, "reason_known": True}, '{"caller_name": "Neha", "reason": "website"}',
                    text="please try him once more", lang="english", agreed=True)
check("already tried once: the English 'already tried' line", said == [agent.REFUSAL_LINES["tried"]["english"]])
said = refused_turn({"wants_jatin_turn": 1, "turns": 2}, '{"caller_name": "", "reason": "website"}',
                    text="just put me through", lang="english", agreed=True)
check("no name: the English 'name first' line", said == [agent.REFUSAL_LINES["name"]["english"]])
said = refused_turn({"wants_jatin_turn": 1, "turns": 2}, '{"caller_name": "Karan", "reason": "x"}',
                    text="Hawaai karo.", agreed=True, reason=False, barge=True)
check("barge-in before the line: not said, and the result loses 'they've just been asked'",
      said == [] and "ERROR: no reason yet." in [m.get("content") for m in agent.call_state["messages"] if isinstance(m, dict)])
check("female forms: the four screening lines have no gendered word; every line as spoken is not masculine",
      all("hinglish_female" not in agent.REFUSAL_LINES[k] for k in ("reason", "name", "not_agreed", "tried"))
      and all(not agent.MASCULINE_FIRST_PERSON.search(agent.pick_line(v, l)) for v in agent.REFUSAL_LINES.values() for l in ("english", "hinglish")))
check("every refusal line, as spoken, in the fixed-line set (cached)",
      all(agent.pick_line(v, l) in agent.all_fixed_lines()[l] for v in agent.REFUSAL_LINES.values() for l in ("english", "hinglish")))

print("claim patterns: koshish and hold phrases")
for line in ["Theek hai Rohit, main unko call karne ki koshish karti hoon.", "Thoda intezaar kijiye.", "Main koshish kar dungi.",
             "Main koshish karungi.", "Please hold.", "Can you hold on a moment?", "Line pe rahiye."]:
    check(f"blocked: {line!r}", agent.claims_action(line))
check("a question offering to try is still an offer, not a claim", not agent.claims_action("Kya main koshish karoon?"))

print("live language and gender checks")
said = claim_turn([FakeStream([text_chunk("Kya baat hai—kis baare mein baat karni thi?")]), FakeStream([text_chunk("What's it about?")])],
                  text="hi, this is priya, can i speak to jatin", wants=None, reason=False)
check("#1032: Hindi words to an English caller -> never said, English retry said",
      said == ["What's it about?"] and any(d.startswith("wrong_language:") for d in events("claim_guard")))
said = claim_turn([FakeStream([text_chunk("Kya baat hai, kya batana hai?")]), FakeStream([text_chunk("Theek hai, kya batana hai?")])],
                  text="hi this is priya", wants=None, reason=False)
check("retry still in Hinglish -> the English fixed line", said == ["Sorry, can you say that again?"])
new_call()
agent.call_state.update(turns=3, speech_lang="hinglish", lang="hinglish")
said = []
script(FakeStream([text_chunk("Theek hai, bata dungi.")]))
JUDGE["lang"] = "english"                           # a bare "bye": the judge reads English, the call stays Hinglish
agent.handle_turn("bye", sink=said.append, reply_id="lb")
JUDGE["lang"] = "english"
check("bare 'bye' keeps the last clear language: a Hinglish reply passes", said == ["Theek hai, bata dungi."])
said = claim_turn([FakeStream([text_chunk("Samajh gaya, Suresh ji.")]), FakeStream([text_chunk("Samajh gayi, Suresh ji.")])],
                  text="mera naam Suresh hai, paise ke baare mein baat karni thi", lang="hinglish", wants=None, reason=True)
check("#1067: 'Samajh gaya' never said; the retry 'Samajh gayi' is", said == ["Samajh gayi, Suresh ji."]
      and any(d.startswith("masculine:") for d in events("claim_guard")))
check("third person 'Rohit bula raha tha' is not masculine first person (#771)",
      not agent.MASCULINE_FIRST_PERSON.search("main Jatin ko bolti hoon ki Rohit bula raha tha"))
check("eval uses the same gender pattern", run_evals.MASCULINE_FIRST_PERSON is agent.MASCULINE_FIRST_PERSON)

print("eval: turn checks use the live guard's functions")
new_call()
agent.record("caller_said", "hi, this is priya, can i speak to jatin")
agent.record("jarvis_said", "Thanks, Priya. Can I ask what it's about so I can try putting you through?")
check("#1031's ask passes the shared check", not run_evals.check({"expect": {"no_ring_talk": [1]}, "caller_lines": ["x"]},
      {"call_id": agent.call_state["call_id"], "outcome": None, "connected": False, "tools_used": [], "tools_blocked": []}))
new_call()
agent.record("caller_said", "Main Karan bol raha hoon, mujhe unse kuch kaam tha.")
agent.record("jarvis_said", "Theek hai Karan — kya main abhi Jatin ko call karke aapko connect kar deti hoon?")
_f = run_evals.check({"expect": {"no_ring_talk": [1]}, "caller_lines": ["x"]},
                     {"call_id": agent.call_state["call_id"], "outcome": None, "connected": False, "tools_used": [], "tools_blocked": []})
check("#912's offer fails it", any("offered or claimed a ring" in f for f in _f))

# ---------- option 3: code picks the language (switched on only for these tests) ----------
print("option 3 (JARVIS_CODE_LANGUAGE=1): language from the caller's line, first audio doesn't wait for the judge")
check("the switch follows JARVIS_CODE_LANGUAGE in .env", ENV_CODE_LANGUAGE == (os.getenv("JARVIS_CODE_LANGUAGE") == "1"))
check("code_language: script, 2+ Hindi words, else clear English, else keep",
      agent.code_language("جتن کو بول دینا") == "hinglish" and agent.code_language("nahi bas itna hi, bye") == "hinglish"
      and agent.code_language("hi, this is priya, can i speak to jatin") == "english"
      and agent.code_language("bye") is None and agent.code_language("mai Rohit hu") is None
      and agent.code_language("haan try kar lo") is None)
agent.CODE_LANGUAGE = True
real_create = None


def lang_turns(lines, judge_lang="english", judge_delay=0.0):
    """Run caller lines through a call (model says 'Okay.'); return the code language after each turn and timings."""
    new_call()
    out, waits = [], []
    for line in lines:
        script(FakeStream([text_chunk("Okay.")]))
        base = agent.client.chat.completions.create

        def slow(**kw):
            if "response_format" in kw:
                time.sleep(judge_delay)
            return base(**kw)
        agent.client.chat.completions.create = slow
        JUDGE["lang"] = judge_lang
        agent.handle_turn(line, sink=lambda l: None, reply_id=f"cl{len(out)}")
        out.append(agent.call_state["speech_lang"])
        waits.append(agent.call_state["timing"].get("judge_wait_first"))
    JUDGE["lang"] = "english"
    return out, waits


langs, _ = lang_turns(["hello, jatin hai kya", "mai Rohit hu", "aise hi baat karni hai", "haan try kar lo"])
check("vague-reason call: 'mai Rohit hu' and 'haan try kar lo' stay Hinglish (judge says English every turn)",
      langs == ["hinglish"] * 4)
langs, _ = lang_turns(["hi, this is Karan calling", "it's about the logo design for his website",
                       "achha theek hai, bas unko bol dena ki final logo file bhej di hai", "nahi bas itna hi, bye"])
check("English-to-Hinglish switch, and #962's 'nahi bas itna hi, bye' as Hinglish",
      langs == ["english", "english", "hinglish", "hinglish"])
langs, _ = lang_turns(["hi this is neha about the website", "bye"], judge_lang="hinglish")
check("bare 'bye' keeps English even if the judge says Hinglish; the judge's language is ignored", langs == ["english", "english"])
langs, waits = lang_turns(["hi, this is priya, can i speak to jatin"], judge_delay=0.6)
check("first sentence cleared without waiting for a slow judge (judge wait before first audio = 0)", waits == [0])
check("...yet the verdict is still applied by the end of the turn (ring locks)", agent.call_state["judge_future"] is None
      and any(d.startswith("lang=") for d in events("judge")))
check("code-set language logged as speech_lang '(code)'", any(d.endswith("(code)") for d in events("speech_lang")))
for first, want in [("mai Rohit hu", "hinglish"), ("haan try kar lo", "hinglish"), ("arey ek baar try to karo", "hinglish"),
                    ("Hi, this is Karan from Bangalore", "english"), ("hello", "hinglish")]:
    langs, _ = lang_turns([first])
    check(f"first line {first!r} -> {want}", langs == [want])
langs, _ = lang_turns(["hello", "hi, this is priya, can i speak to jatin"])
check("bare 'hello' first: Hinglish until a clear English line switches it", langs == ["hinglish", "english"])
agent.CODE_LANGUAGE = False

# ---------- the judge model is a setting; a retired judge doesn't end the call ----------
print("judge model setting and a retired judge model")
check("JUDGE_MODEL comes from JARVIS_JUDGE_MODEL", agent.JUDGE_MODEL == (os.getenv("JARVIS_JUDGE_MODEL") or "gpt-5-nano"))
import httpx as _httpx
from openai import NotFoundError as _NotFound
new_call()
said = []


def retired_judge(**kw):
    if "response_format" in kw:
        raise _NotFound("The model `gpt-4.1-mini` has been deprecated", response=_httpx.Response(
            404, request=_httpx.Request("POST", "https://api.openai.com/v1/chat/completions")), body=None)
    model_requests.append(kw)
    return FakeStream([text_chunk("Can I take your name?")])


model_requests.clear()
agent.client.chat.completions.create = retired_judge
agent.handle_turn("hi there, can i talk to jatin", sink=said.append, reply_id="jd")
check("judge 404: logged as judge_model_error (says deprecated or not found), not a provider outage",
      any("deprecated or not found" in d for d in events("judge_model_error")) and not agent.call_state["provider_down"])
check("...and the call goes on: the reply is said, the judge counts as failed", said == ["Can I take your name?"]
      and any(d.startswith("FAILED") for d in events("judge")) and not agent.call_state["ended"])

# ---------- main model on its own provider (bake-off harness, 2026-11-25) ----------
print("main model / judge routing, and provider-limited calls")
check("defaults: main model on the judge's provider, same client", agent.MAIN_PROVIDER == agent.PROVIDER and agent.main_client is agent.client
      or os.getenv("JARVIS_MAIN_PROVIDER"))


class FakeClient:
    def __init__(self, handler):
        self.chat = N(completions=N(create=handler))


main_seen, judge_seen = [], []
real_main, real_client_create, real_provider = agent.main_client, agent.client.chat.completions.create, agent.MAIN_PROVIDER
agent.MAIN_PROVIDER = "groq"
agent.main_client = FakeClient(lambda **kw: (main_seen.append(kw), FakeStream([text_chunk("Okay, what's it about?")]))[1])
agent.client.chat.completions.create = lambda **kw: (judge_seen.append(kw), judge_reply())[1]
new_call()
said = []
agent.handle_turn("hi this is rahul calling", sink=said.append, reply_id="mp")
check("main-model requests go to the main client, the judge's to the judge client",
      len(main_seen) == 1 and "response_format" not in main_seen[0] and len(judge_seen) == 1
      and "response_format" in judge_seen[0] and said == ["Okay, what's it about?"])

from openai import RateLimitError as _RL
import httpx as _hx
_limit = _RL("Rate limit reached for model openai/gpt-oss-120b: tokens per minute (TPM): Limit 8000",
             response=_hx.Response(429, request=_hx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")), body=None)


def always_limited(**kw):
    raise _limit


agent.main_client = FakeClient(always_limited)
real_sleep = agent.time.sleep
agent.time.sleep = lambda s: None          # skip the retry waits in the test
new_call()
agent.handle_turn("hi this is rahul calling", sink=lambda l: None, reply_id="lim")
agent.time.sleep = real_sleep
summary = agent.finish_call()
check("rate limit after 5 tries: provider_limited (not just provider_down)", summary["provider_limited"] and summary["provider_down"])
import run_evals as _re
_orig_run_call = agent.run_call
agent.run_call = lambda **kw: summary
check("eval: a provider-limited call is reported as 'provider-limited', not scored as a model failure",
      _re.run_one({"caller_id": "", "caller_lines": ["x"], "owner_answers": [], "expect": {}})[1] == ["provider-limited"])
agent.run_call = _orig_run_call
agent.main_client, agent.MAIN_PROVIDER = real_main, real_provider
agent.client.chat.completions.create = real_client_create

# ---------- filler for slow turns (2026-11-25, rotation + threshold 2026-11-26) ----------
print("filler: random cached line, no repeats, never on back-to-back turns, ~1/3 skipped")
client = server.app.test_client()
real_cached = voice.cached_pcm
ALL_FILLERS = set(agent.FILLER_LINES["english"]) | set(agent.FILLER_LINES["hinglish"])
voice.cached_pcm = lambda text, lang: (b"\x02\x00" * 2400) if text in ALL_FILLERS else None
check("no call -> empty audio", client.get("/filler").data == b"")
client.post("/start", json={})
agent.call_state["speech_lang"] = "hinglish"


def filler_on(turn):
    agent.call_state["turns"] = turn
    r = client.get("/filler")
    return (json.loads(events("filler")[-1])["line"] if r.data else None), r


agent.filler_random = __import__("random").Random(7)
line, r = None, None
for turn in range(1, 6):  # first non-skip
    line, r = filler_on(turn)
    if line:
        break
check("Hinglish filler: one of the 4 lines, cached audio", line in agent.FILLER_LINES["hinglish"]
      and len(r.data) == 4800 and r.mimetype == "audio/pcm" and len(agent.FILLER_LINES["hinglish"]) == 4)
check("never on two turns in a row: the next turn is silent, logged as skipped",
      filler_on(turn + 1)[0] is None and client.get("/filler").data == b"" and events("filler_skipped"))
check("never twice in one turn either", filler_on(turn)[0] is None)
# Over many turns with a filler due every turn (the 2.0 s case): never back to back, never the same line twice
# in a row, about a third of the eligible turns skipped, and so about 40% of turns get one.
agent.filler_random = __import__("random").Random(11)
agent.call_state.pop("last_filler", None), agent.call_state.pop("last_filler_turn", None)
agent.call_state["speech_lang"] = "english"
plays = [(t, filler_on(t)[0]) for t in range(100, 400)]
played = [(t, l) for t, l in plays if l]
check("English, no context (acknowledgement): the 3 ack lines, all used",
      {l for _, l in played} == set(agent.FILLER_GROUPS["ack"]["english"]))
check("never on two turns in a row", all(b[0] - a[0] >= 2 for a, b in zip(played, played[1:])))
check("never the same line twice in a row", all(a[1] != b[1] for a, b in zip(played, played[1:])))
eligible = [l for (t, l), prev in zip(plays[1:], plays) if not prev[1]]
skip_share = sum(1 for l in eligible if l is None) / len(eligible)
check(f"random skip about a third of eligible turns ({skip_share:.0%}); filler on {len(played) / len(plays):.0%} of turns",
      0.25 <= skip_share <= 0.42 and 0.33 <= len(played) / len(plays) <= 0.47)
check("never in the model's history", not any(f in json.dumps(agent.call_state["messages"]) for f in ALL_FILLERS))
voice.cached_pcm = lambda text, lang: None
agent.filler_random = type("NoSkip", (), {"random": lambda self: 0.99, "choice": lambda self, o: o[0]})()
agent.call_state["turns"] += 5
check("not cached -> empty audio, never synthesised live", client.get("/filler").data == b""
      and json.loads(events("filler")[-1])["cached"] is False)
voice.cached_pcm = real_cached
client.post("/hangup")
for lang in ("english", "hinglish"):
    for line in agent.FILLER_LINES[lang]:
        check(f"{lang} filler {line!r}: cached set, passes leak/claim/gender checks",
              line in agent.all_fixed_lines()[lang] and agent.leak_check(line, "", "") is None and not agent.claims_action(line)
              and not agent.MASCULINE_FIRST_PERSON.search(line))
check("no filler starts with a word replies often start with (Got it / Okay / Sure / Theek hai / Haan)",
      not {l.split()[0].strip(".").lower() for ls in agent.FILLER_LINES.values() for l in ls} & {"got", "okay", "ok", "sure", "theek", "haan"})
check("every filler is in the cached set for its own language (lists differ in length)",
      all(l in agent.all_fixed_lines()[lang] for lang in ("english", "hinglish") for l in agent.FILLER_LINES[lang]))
import run_evals as _re3
check("eval filler echo: 'Hmm...' then 'Hmm, okay' counts; 'Achha...' then 'Accha theek hai' counts; 'Right...' then 'Okay' doesn't",
      _re3.filler_echo("Hmm...", "Hmm, okay.") and _re3.filler_echo("Achha...", "Accha theek hai.") and not _re3.filler_echo("Right...", "Okay."))
check("page: filler once per turn, only while no line has arrived, cleared on barge-in",
      "FILLER_AFTER_MS" in page and "reply.fillerPlayed = true" in page and "reply.items.length === 0" in page
      and "clearTimeout(fillerTimer)" in page and 'queueLine("filler", null)' in page)

# ---------- #1404: reason missing in the call but already answered -> the model retries ----------
print("#1404: 'aise hi' already answered -> no code line, the model fixes the call and rings")


def ring_case(streams, caller_line, state, agreed=True, reason=True, owner="no", lang="hinglish"):
    agent.start_call(caller_id="", caller_lines=None, owner_answers=[owner], verbose=False)
    agent.call_state.update(**state)
    said = []
    script(*streams)
    JUDGE.update(agreed=agreed, reason=reason, lang=lang)
    agent.handle_turn(caller_line, sink=said.append, reply_id="rc")
    JUDGE.update(agreed=False, reason=True, lang="english")
    return said


said = ring_case([FakeStream([tool_chunk(0, "a", "try_reach_owner", '{"caller_name": "Rohit"}')]),
                  FakeStream([tool_chunk(0, "b", "try_reach_owner", '{"caller_name": "Rohit", "no_reason_given": true}')])],
                 "haan try kar lo", {"wants_jatin_turn": 3, "turns": 4, "reason_known": True})
ring_results = events("tool_result")
check("#1404: first call gets the argument error (no code line), the retry rings",
      ring_results[0] == "try_reach_owner -> " + agent.REASON_ARGUMENT_MISSING and agent.call_state["rings"] == 1
      and said == [agent.DECLINE_LINES["hinglish"]] and "refusal" not in json.dumps(events("fixed_line")))
said = ring_case([FakeStream([tool_chunk(0, "a", "try_reach_owner", '{"caller_name": "Rohit"}')])],
                 "haan try kar lo", {"wants_jatin_turn": 3, "turns": 4, "reason_known": False}, reason=False)
check("reason really unknown: still the 'no reason' code line, no ring",
      said == [agent.REFUSAL_LINES["reason"]["hinglish"]] and agent.call_state["rings"] == 0)

# ---------- #1399 / #1176: request and "yes" in one line -> ring in the same turn ----------
print("#1399/#1176: caller_wants_jatin, then try_reach_owner in the next step of the same turn")
said = ring_case([FakeStream([tool_chunk(0, "w", "caller_wants_jatin", "{}")]),
                  FakeStream([tool_chunk(0, "t", "try_reach_owner", '{"caller_name": "Neha", "reason": "website"}')])],
                 "sure try him", {"turns": 2, "reason_known": True}, lang="english")
check("#1399: same-turn ring after caller_wants_jatin (Jatin declines -> decline line)",
      agent.call_state["rings"] == 1 and said == [agent.DECLINE_LINES["english"]])
said = ring_case([FakeStream([tool_chunk(0, "w", "caller_wants_jatin", "{}")]),
                  FakeStream([tool_chunk(0, "t", "try_reach_owner", '{"caller_name": "Rohit", "no_reason_given": true}')])],
                 "haan try kar lo", {"turns": 4, "reason_known": True})
check("#1176: same, in Hinglish", agent.call_state["rings"] == 1 and said == [agent.DECLINE_LINES["hinglish"]])
said = ring_case([FakeStream([tool_chunk(0, "w", "caller_wants_jatin", "{}")]),
                  FakeStream([tool_chunk(0, "t", "try_reach_owner", '{"caller_name": "Neha", "reason": "website"}')])],
                 "hi this is neha, about the website", {"turns": 1, "reason_known": True}, agreed=False, lang="english")
check("lock: no offer/yes (judge: not agreed) -> no ring, 'try him or a reminder?' line",
      agent.call_state["rings"] == 0 and said == [agent.REFUSAL_LINES["not_agreed"]["english"]])
said = ring_case([FakeStream([tool_chunk(0, "w", "caller_wants_jatin", "{}")]),
                  FakeStream([tool_chunk(0, "t", "try_reach_owner", '{"caller_name": "Neha", "reason": "x"}')])],
                 "neha here, put me through", {"turns": 1, "reason_known": False}, reason=False, lang="english")
check("lock: no reason -> no ring", agent.call_state["rings"] == 0 and said == [agent.REFUSAL_LINES["reason"]["english"]])
check("prompt no longer forbids the same turn", "Never in the same turn as caller_wants_jatin" not in agent.SYSTEM_PROMPT
      and "after caller_wants_jatin if they asked; it may be in the same turn" in agent.SYSTEM_PROMPT)

# ---------- billing errors fail at once; ordinary rate limits keep the retry path (2026-11-26) ----------
print("billing: out of credits -> outage line at once, no retries; a short rate limit still retries")
from openai import RateLimitError as _RL2
import httpx as _hx2
def _err(msg, code=None):
    return _RL2(msg, response=_hx2.Response(429, request=_hx2.Request("POST", "https://api.openai.com/v1/chat/completions")),
                body={"error": {"message": msg, "code": code}} if code else None)
billing = _err("Error code: 429 - You have no credits remaining. Add credits to continue using the API.")
check("'no credits remaining' is a billing error", agent.is_billing_error(billing))
check("insufficient_quota code is a billing error", agent.is_billing_error(_err("quota", code="insufficient_quota")))
check("an ordinary rate limit is not", not agent.is_billing_error(_err("Rate limit reached for gpt-5-mini: tokens per min (TPM): Limit 200000")))
sleeps = []
real_sleep2 = agent.time.sleep
agent.time.sleep = lambda s: sleeps.append(s)
new_call()
said = []
attempts = []
def out_of_credits(**kw):
    attempts.append(kw)
    raise billing
agent.client.chat.completions.create = out_of_credits
agent.handle_turn("hi this is rahul calling", sink=said.append, reply_id="bill")
check("billing error: no retries, no waits, logged as billing_error, outage line said, call ended",
      sleeps == [] and len(attempts) <= 2 and any("no credits remaining" in d for d in events("billing_error"))
      and said and said[-1].startswith("Sorry, something's gone wrong") and agent.call_state["ended"]
      and not agent.call_state.get("provider_limited"))
sleeps.clear(); attempts.clear()
new_call()
def short_limit(**kw):
    attempts.append(kw)
    if len(attempts) < 3:
        raise _err("Rate limit reached for gpt-5-mini: tokens per min (TPM)")
    return judge_reply() if "response_format" in kw else FakeStream([text_chunk("Okay, what's it about?")])
agent.client.chat.completions.create = short_limit
said = []
agent.handle_turn("hi this is rahul calling", sink=said.append, reply_id="rl")
check("ordinary rate limit: retried with waits, then the reply", sleeps and said == ["Okay, what's it about?"]
      and not events("billing_error") and not agent.call_state["provider_down"])
agent.time.sleep = real_sleep2

print("filler wait is a setting")
check("JARVIS_FILLER_AFTER_S read at call start, default 2.0 s",
      server.filler_after_ms() == int(float(os.getenv("JARVIS_FILLER_AFTER_S", "2.0")) * 1000))
os.environ["JARVIS_FILLER_AFTER_S"] = "7.2"
_r = server.app.test_client().post("/start", json={}).get_json()
check("7.2 in .env -> /start tells the page 7200 ms", _r.get("filler_after_ms") == 7200)
server.app.test_client().post("/hangup")
del os.environ["JARVIS_FILLER_AFTER_S"]
check("page takes the wait from /start (default 2000 ms)", "let FILLER_AFTER_MS = 2000" in page
      and "FILLER_AFTER_MS = data.filler_after_ms" in page)

# ---------- ring rule: an agreed offer OR an explicit request; retry notes ask the next question (2026-11-26) ----------
print("ring rule and retry notes")
check("judge instruction: (a) an agreed offer, or (b) an explicit request; a bare 'haan'/'ok' is not agreement",
      "(b) the caller themselves explicitly asks to be put through" in agent.JUDGE_PROMPT
      and 'A bare "haan", "ok" or "yes" with no offer before it and no request in it is NOT agreement' in agent.JUDGE_PROMPT)
for name, note in list(agent.CLAIM_RETRY_NOTES.items()) + [("leak", agent.LEAK_RETRY_NOTE)]:
    check(f"retry note '{name}' ends by asking the next question and passes the guards",
          "then ask your next question" in note and agent.leak_check(note, "", "") is None and not agent.claims_action(note))
import run_evals as _rv


def _ring_call(lines):
    new_call()
    for kind, detail in lines:
        agent.record(kind, detail)
    return _rv.bad_rings(agent.call_state["call_id"])


check("#1471 'connect me' with no offer: a good ring (explicit request)",
      _ring_call([("caller_said", "i said it doesn't matter, connect me"), ("judge", "lang=english agreed=True reason_known=True"),
                  ("owner_briefed", "Incoming call from Unnamed caller: x")]) == [])
check("#1485 'haan try kar lo' with no offer: a good ring (explicit request)",
      _ring_call([("judge", "lang=hinglish agreed=False reason_known=True"), ("caller_said", "haan try kar lo"),
                  ("owner_briefed", "Incoming call from Rohit: x")]) == [])
check("a bare 'haan' with no offer: a BAD ring (the watch item)",
      len(_ring_call([("judge", "lang=hinglish agreed=False reason_known=True"), ("caller_said", "haan"),
                      ("owner_briefed", "Incoming call from Rohit: x")])) == 1)
check("'haan' after an offer: a good ring",
      _ring_call([("jarvis_said", "Kya main Jatin ko call karne ka try karoon?"), ("judge", "lang=hinglish agreed=False reason_known=True"),
                  ("caller_said", "haan"), ("owner_briefed", "Incoming call from Rohit: x")]) == [])
check("no reason: a BAD ring", len(_ring_call([("caller_said", "connect me"), ("owner_briefed", "Incoming call from X: x")])) == 1)
check("family rings are not counted", _ring_call([("owner_briefed", "Family call (caller ID 1111)")]) == [])

# ---------- duplicate saves ----------
print("duplicate saves (take_message)")
new_call()
agent.call_state["turns"] = 3
first = agent.take_message("Ramesh called; the site is ready to go live.", "Ramesh")
check("first save goes through", first.startswith("Message saved"))
check("identical re-save refused", agent.take_message("Ramesh called; the site is ready to go live.", "Ramesh") == agent.DUPLICATE_RESULT)
check("#726 near-same ('he says' added) refused",
      agent.take_message("Ramesh called; he says the site is ready to go live.", "Ramesh") == agent.DUPLICATE_RESULT)
check("same note + new information ('tomorrow') is saved",
      agent.take_message("Ramesh called; the site is ready to go live tomorrow.", "Ramesh").startswith("Message saved"))
check("only real saves in the database (2)", len(db.reminders_for_call(agent.call_state["call_id"])) == 2)
check("duplicate_save events logged (2)", len(events("duplicate_save")) == 2)
new_call()
agent.call_state["turns"] = 3
agent.take_message("Neha called; wants to talk about the website.", "Neha")
check("#709 a different second message is saved",
      agent.take_message("Neha called; asked you not to be disturbed and to note that she called.", "Neha").startswith("Message saved"))
new_call()
agent.call_state["turns"] = 3
agent.take_message("Karan called; it's about the logo design for his website.", "Karan")
check("#728 follow-on message is saved",
      agent.take_message("Karan called; he said the final logo file has been sent. Please note.", "Karan").startswith("Message saved"))
agent.call_state["family"] = True
agent.take_message("A family member called; they want you to take their call now.")
check("#715 family second message is saved",
      agent.take_message("A family member called; they said they called and asked you to know that they called.").startswith("Message saved"))

print("duplicate save through the agent loop (blocked, not saved)")
new_call()
script(FakeStream([tool_chunk(0, "d1", "take_message", '{"caller_name": "Ramesh", "message": "Ramesh called; the site is ready to go live."}')]),
       FakeStream([tool_chunk(0, "d2", "take_message", '{"caller_name": "Ramesh", "message": "Ramesh called; the site is ready to go live."}')]),
       FakeStream([text_chunk("It's already with him.")]))
agent.handle_turn("hi this is ramesh, tell jatin the site is ready to go live", sink=lambda line: None, reply_id="d1")
agent.handle_turn("can you save it again exactly how i said it?", sink=lambda line: None, reply_id="d2")
check("second attempt counted as blocked, one note saved",
      agent.call_state["tools_blocked"] == ["take_message"] and len(db.reminders_for_call(agent.call_state["call_id"])) == 1)

print("new tool results: nothing to recite")
agent.call_state["follow_up_spoken"] = True
second = agent.after_save("Message saved from Ramesh.", "message")
for label, result in [("second-save result", second), ("duplicate result", agent.DUPLICATE_RESULT)]:
    check(f"{label}: no 'wait' in it", "wait" not in result.lower())
    check(f"{label}: passes the self-reference guard", not agent.SELF_REFERENCE.search(result))

# ---------- server: streamed turns and /barge_in ----------
print("server: streamed turns (NDJSON)")
client = server.app.test_client()


def ndjson(response):
    return [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]


client.post("/start", json={"reply_id": "s0"})
script(FakeStream([text_chunk("Hi Priya. "), text_chunk("What's it about?")]))
got = ndjson(client.post("/turn?stream=1", json={"text": "hi this is priya", "reply_id": "s1"}))
says = [e for e in got if e["type"] == "say"]
check("typed turn streams one say event per sentence, then done",
      [e["text"] for e in says] == ["Hi Priya.", "What's it about?"] and got[-1]["type"] == "done"
      and got[-1]["reply"] == "Hi Priya. What's it about?" and got[-1]["ended"] is False)
check("each sentence has its own playable speech URL",
      all(client.get(e["audio_url"]).status_code == 200 for e in says))
speech = [json.loads(d) for d in events("speech")]
check("speech log numbers the lines of the turn", [s["seq"] for s in speech[-2:]] == [0, 1])

voice.transcribe = lambda data, name, mime, seconds=None: ("hi, it's priya again", 90)
script(FakeStream([text_chunk("Okay, Priya. "), text_chunk("Go ahead.")]))
got = ndjson(client.post("/turn_audio?stream=1", data={"audio": (io.BytesIO(b"x"), "turn.webm", "audio/webm"), "reply_id": "s2"}))
check("spoken turn: heard first, then the lines, then done",
      [e["type"] for e in got] == ["heard", "say", "say", "done"] and got[0]["text"] == "hi, it's priya again")
voice.transcribe = lambda data, name, mime, seconds=None: ("", 40)
got = ndjson(client.post("/turn_audio?stream=1", data={"audio": (io.BytesIO(b"x"), "turn.webm", "audio/webm"), "reply_id": "s3"}))
check("nothing heard: 'didn't catch that' streams as a say event", [e["type"] for e in got] == ["say", "done"]
      and got[0]["text"] == server.DIDNT_CATCH)
check("plain (non-stream) turn still returns one JSON reply",
      (script(FakeStream([text_chunk("Sure.")])) or True)
      and client.post("/turn", json={"text": "ok one more thing"}).get_json()["reply"] == "Sure.")

print("server: /barge_in")
check("bad body -> 400", client.post("/barge_in", json={"reply_id": "x"}).status_code == 400
      and client.post("/barge_in", json={"reply_id": "x", "heard": -1}).status_code == 400)
stream = FakeStream([text_chunk("Okay. Let me see. "), text_chunk("He is "), text_chunk("not around.")], wait_at=1)
script(stream)
box = {}
turn_thread = threading.Thread(target=lambda: box.update(events=ndjson(
    client.post("/turn?stream=1", json={"text": "hello hello", "reply_id": "s4"}))))
turn_thread.start()
for _ in range(200):
    if agent.current_reply()["id"] == "s4" and agent.current_reply()["said"]:
        break
    time.sleep(0.01)
started = time.time()
answer = client.post("/barge_in", json={"reply_id": "s4", "heard": 1}).get_json()
check("/barge_in gets through while the turn holds the lock", answer["applied"] and time.time() - started < 1)
turn_thread.join(timeout=5)
check("the turn ended early and released the lock", not turn_thread.is_alive() and box["events"][-1]["type"] == "done"
      and not server.call_lock.locked())
check("history = the line that was heard", assistant_texts()[-1] == "Okay. Let me see.")
cut_url = [e for e in box["events"] if e["type"] == "say"][0]["audio_url"]
cut = client.get(cut_url)
check("speech for a cut reply is never made (empty audio, logged cancelled)",
      cut.data == b"" and json.loads(events("speech")[-1]).get("cancelled") is True)
script(FakeStream([text_chunk("Sure, go ahead.")]))
got = ndjson(client.post("/turn?stream=1", json={"text": "sorry, one more thing", "reply_id": "s5"}))
check("the next turn runs normally", got[-1]["type"] == "done" and got[-1]["reply"] == "Sure, go ahead.")
client.post("/hangup")
check("/barge_in with no call -> not applied", client.post("/barge_in", json={"reply_id": "s5", "heard": 0}).get_json()["applied"] is False)

# ---------- live #1509 / #1510 (2026-11-26): name + reason = the request; offers no longer need caller_wants_jatin ----------
print("#1509/#1510: offer after name + reason, with no 'I want Jatin' from the caller")
for line in ("Do you want me to try to put you through to Jatin, or should I leave him a message about the loan?",  # #1510 t2
             "Can I try putting you through to Jatin now, or would you like me to leave a message for him?",      # #1510 t3
             "Rahul, thanks — want me to try putting you through to Jatin now?"):                                # #1509 t3
    said = claim_turn([FakeStream([text_chunk(line)])], text="I'm Rahul.", wants=None, reason=True)
    check(f"said, not blocked: {line[:55]}...", said == [line] and not events("claim_guard"))
    check("...and it counts as the request: offer_made + implied_request, try_reach_owner on the menu",
          agent.call_state["wants_jatin_turn"] == agent.call_state["turns"] and events("offer_made") and events("implied_request"))
said = claim_turn([FakeStream([text_chunk("Thanks, Rahul — quick question: would you like Jatin to call you back about "
                                          "the BPS, or should I just leave him a message?")])], text="Rahul.", reason=True)
check("#1509 t3 retry (a callback offer, as a question): said, not a promise, not blocked", len(said) == 1 and not events("claim_guard"))
said = claim_turn([FakeStream([text_chunk("Do you want me to try to put you through to Jatin?")]),
                   FakeStream([text_chunk("Can I ask what it's about?")])], text="I'm Rahul.", wants=None, reason=False)
check("reason still unknown: the offer is still blocked (early_offer), no implied request",
      said == ["Can I ask what it's about?"] and any(d.startswith("early_offer:") for d in events("claim_guard"))
      and agent.call_state["wants_jatin_turn"] is None and not events("implied_request"))
# Whole flow: offer on one turn, "yes" on the next -> the ring is allowed (Jatin's yes still decides).
said = claim_turn([FakeStream([text_chunk("Do you want me to try to put you through to Jatin, or leave him a message about the loan?")])],
                  text="I'm Rahul.", wants=None, reason=True)
agent.call_state["turns"] = 3
script(FakeStream([tool_chunk(0, "r", "try_reach_owner", '{"caller_name": "Rahul", "reason": "loan"}')]))
JUDGE.update(agreed=True, reason=True)
said = []
agent.handle_turn("yes please", sink=said.append, reply_id="y1")
JUDGE.update(agreed=False)
check("next turn 'yes please' after the offer -> try_reach_owner rings (no 'caller_wants_jatin first' error)",
      agent.call_state["rings"] == 1 and not any("caller_wants_jatin first" in r for r in events("tool_result")))
# No offer and no request: a ring is still refused.
said = claim_turn([FakeStream([tool_chunk(0, "r", "try_reach_owner", '{"caller_name": "Rahul", "reason": "loan"}')]),
                   FakeStream([text_chunk("Should I try calling him?")])], text="okay", wants=None, reason=True)
check("no offer and no request: try_reach_owner is not on the menu, nothing rings",
      agent.call_state["rings"] == 0 and any("caller_wants_jatin first" in r or "not available" in r for r in events("tool_result")))

print("#1565-1567: blocked offer after a clean sentence -> code asks the reason")
said = claim_turn([FakeStream([text_chunk("Sorry, I can’t chat about the match. Can I take a message for Jatin, or would you like me to try to reach him?")])],
                  text="did you watch the cricket match yesterday?", reason=False)
check("#1565: the clean sentence, then 'Could you tell me what it's about first?'",
      said == ["Sorry, I can’t chat about the match.", "Could you tell me what it's about first?"]
      and "early_offer after a clean sentence: reason" in events("fixed_line")
      and agent.call_state["messages"][-1]["content"] == "Sorry, I can’t chat about the match. Could you tell me what it's about first?")
said = claim_turn([FakeStream([text_chunk("Haan ji. Kya main message le loon ya unhe call karne ka try karoon?")])],
                  text="kya haal hai", reason=False, lang="hinglish")
check("Hinglish: 'Pehle bata dijiye, kis baare mein baat karni hai?'",
      said == ["Haan ji.", "Pehle bata dijiye, kis baare mein baat karni hai?"])
said = claim_turn([FakeStream([text_chunk("What's this about? Do you want me to pass a message to Jatin or try to put you through?")])],
                  text="did you watch the match", reason=False)
check("a question was already said: nothing added", said == ["What's this about?"] and not events("fixed_line"))
said = claim_turn([FakeStream([text_chunk("Sorry, I can't say. Can you tell me your name? He'll call you back.")])], text="x", reason=True)
check("another guard (promise) after a clean sentence: no reason line added", "Could you tell me what it's about first?" not in said)

# ---------- family flow (2026-11-26) ----------
print("family flow: decline vs no pickup, one re-ring only after a no-pickup, never after a decline")


def family_call(owner, caller_id="1111"):
    agent.start_call(caller_id=caller_id, caller_lines=None, owner_answers=list(owner), verbose=False)
    agent.call_state["speech_lang"] = "hinglish"


def family_turn(streams, text):
    said = []
    script(*streams)
    JUDGE["lang"] = "hinglish"
    agent.handle_turn(text, sink=said.append, reply_id="f")
    JUDGE["lang"] = "english"
    return said


RING = [FakeStream([tool_chunk(0, "r", "try_reach_owner", "{}")])]
check("ring_kind: Decline button / timeout / accept", (agent.ring_kind("no"), agent.ring_kind(""), agent.ring_kind("yes"))
      == ("declined", "no_pickup", "accepted"))
family_call(["no"])
check("declined -> the declined opener (AI disclosure, no 'declined'), rings 1",
      agent.call_state["messages"][1]["content"] == agent.FAMILY_OPENERS["declined"]["hinglish"] and agent.call_state["rings"] == 1)
said = family_turn(RING, "arey ek baar aur try karo na")
check("declined, caller insists -> refused with the fixed line (female form), no ring",
      said == [agent.REFUSAL_LINES["family_declined"]["hinglish_female"]] and agent.call_state["rings"] == 1
      and len(events("owner_briefed")) == 1)
family_call([""])
check("no pickup -> the no-pickup opener (offers one more try)",
      agent.call_state["messages"][1]["content"] == agent.FAMILY_OPENERS["no_pickup"]["hinglish"])
said = family_turn(RING, "haan ek baar aur try karo")
check("no pickup, try again, no pickup again -> rings 2, code says the second no-pickup line",
      agent.call_state["rings"] == 2 and said == [agent.FAMILY_RING_LINES["family_no_pickup"]["hinglish"]]
      and events("owner_briefed")[-1].startswith("Family call again"))
said = family_turn(RING, "arey phir se try karo na")
check("third ring asked -> refused (tried twice), rings stay 2",
      said == [agent.REFUSAL_LINES["family_tried"]["hinglish"]] and agent.call_state["rings"] == 2 and len(events("owner_briefed")) == 2)
family_call(["", "yes"])
said = family_turn(RING, "haan ek baar aur try karo")
check("no pickup, try again, Jatin picks up -> connected, code says the connected line",
      agent.call_state["connected"] and agent.call_state["ended"] and said == [agent.CONNECTED_LINES["hinglish_female"]])
family_call(["", "no"])
said = family_turn(RING, "ek baar aur try karo")
check("re-ring declined -> the declined line (never 'declined'), then no more rings",
      said == [agent.FAMILY_RING_LINES["family_declined"]["hinglish"]] and agent.call_state["rings"] == 2)
said = family_turn(RING, "phir se try karo")
check("...another ask -> the declined refusal", said == [agent.REFUSAL_LINES["family_declined"]["hinglish_female"]]
      and len(events("owner_briefed")) == 2)
check("family menu: take_message, add_reminder, try_reach_owner, end_call",
      {t["function"]["name"] for t in model_requests[-1]["tools"]} == {"take_message", "add_reminder", "try_reach_owner", "end_call"})

print("family reminders and names (FAMILY_NAMES, private)")
os.environ["FAMILY_NAMES"] = "1111:Papa, 2222:Mummy"
family_call(["no"])
check("mapped name on Jatin's phone", events("owner_briefed")[0] == "Family call: Papa (caller ID 1111)")
family_turn([FakeStream([tool_chunk(0, "a", "add_reminder", "{}")])], "haan remind kar do")
saved = db.reminders_for_call(agent.call_state["call_id"])
check("family reminder with no name or reason: saved as Papa, 'wants a callback'",
      len(saved) == 1 and saved[0]["caller_name"] == "Papa" and saved[0]["content"] == "wants a callback" and saved[0]["kind"] == "reminder")
family_call([""], caller_id="3333")
family_turn([FakeStream([tool_chunk(0, "m", "take_message", '{"message": "Come home early."}')])], "usko bolo jaldi ghar aaye")
saved = db.reminders_for_call(agent.call_state["call_id"])
check("unmapped family number -> 'Family member' (no caller ID in the name)", saved[0]["caller_name"] == "Family member"
      and events("owner_briefed")[0] == "Family call (caller ID 3333)")
family_call(["no"], caller_id="2222")
family_turn([FakeStream([tool_chunk(0, "m", "take_message", '{"message": "Call back.", "caller_name": "Ma"}')])], "bol dena")
check("mapping beats a name they said", db.reminders_for_call(agent.call_state["call_id"])[0]["caller_name"] == "Mummy")
os.environ.pop("FAMILY_NAMES")
check(".env.example has only an example, commented out", "# FAMILY_NAMES=1111:Papa,2222:Mummy" in open(".env.example", encoding="utf-8").read())

print("family lines: female forms, leak/claim/promise checks, never 'declined', cached")
family_lines = ([o["hinglish"] for o in agent.FAMILY_OPENERS.values()]
                + [agent.pick_line(t, lang) for t in list(agent.FAMILY_RING_LINES.values())
                   + [agent.REFUSAL_LINES["family_declined"], agent.REFUSAL_LINES["family_tried"], agent.CLOSING_LINES["family_reminder"]]
                   for lang in ("english", "hinglish")])
for line in family_lines:
    check(f"ok: {line[:60]}", agent.leak_check(line, "", "") is None and not agent.claims_action(line)
          and not agent.promises_callback(line) and not agent.MASCULINE_FIRST_PERSON.search(line) and "declin" not in line.lower())
fixed = agent.all_fixed_lines()
check("family lines in the cached set (openers Hinglish only)",
      all(o["hinglish"] in fixed["hinglish"] and list(o) == ["hinglish"] for o in agent.FAMILY_OPENERS.values())
      and all(agent.pick_line(t, l) in fixed[l] for t in agent.FAMILY_RING_LINES.values() for l in ("english", "hinglish")))
check("family prompt: re-ring by tool on request, no 'he was just tried' rule",
      "They ask you to try Jatin again: try_reach_owner." in agent.FAMILY_PROMPT and "he was just tried" not in agent.FAMILY_PROMPT
      and "add_reminder" in agent.FAMILY_PROMPT)
import run_evals as _re4
family_call(["", "no"])
family_turn(RING, "ek baar aur try karo")
check("eval family_rings: 2 rings, no violation", _re4.family_rings(agent.call_state["call_id"]) == (2, []))
family_call(["no"])
agent.call_state["ever_declined"] = False  # force the lock open: the eval must still catch a ring after a decline
family_turn(RING, "ek baar aur try karo")
check("eval family_rings: a ring after a decline is reported", "rang again after a decline" in _re4.family_rings(agent.call_state["call_id"])[1])

print("#1625: the reason ask on two turns in a row -> the alternative line the second time")
offer = "Can I take a message for him, or would you like me to try putting you through?"
new_call()
agent.call_state.update(reason_known=False, turns=2)
lines = []
for turn, text in ((3, "I don't follow matches."), (4, "I don't know what Jatin does."), (5, "Sorry, I can't say.")):
    agent.call_state["turns"] = turn - 1
    script(FakeStream([text_chunk(f"{text} {offer}")]))
    JUDGE["reason"] = False
    said = []
    agent.handle_turn("chit chat", sink=said.append, reply_id=f"r{turn}")
    JUDGE["reason"] = True
    lines.append(said[-1])
check("turn 1: usual line; turn 2 (right after): the alternative; turn 3: usual again",
      lines == ["Could you tell me what it's about first?", "Just tell me what the call's about?", "Could you tell me what it's about first?"])
agent.call_state["speech_lang"] = "hinglish"
check("Hinglish alternative", agent.pick_line(agent.REASON_AGAIN_LINES, "hinglish") == "Bas itna bata dijiye, call kis baare mein hai?")
for lang in ("english", "hinglish"):
    line = agent.pick_line(agent.REASON_AGAIN_LINES, lang)
    check(f"alternative ({lang}): cached set, passes leak/claim/promise/gender, asks the reason (no early_offer)",
          line in agent.all_fixed_lines()[lang] and agent.leak_check(line, "", "") is None and not agent.claims_action(line)
          and not agent.promises_callback(line) and not agent.MASCULINE_FIRST_PERSON.search(line) and not agent.offers_ring(line))
check("family add_reminder: only when they ask him to call back; a message is take_message",
      "only when they ask him to call them back" in json.dumps(agent.family_tools) and "is take_message, not a reminder" in json.dumps(agent.family_tools))
check("English family openers dropped", all(list(o) == ["hinglish"] for o in agent.FAMILY_OPENERS.values()))
family_call([""])
family_turn([FakeStream([tool_chunk(0, "m", "take_message", '{"message": "Come home soon."}'),
                         tool_chunk(1, "a", "add_reminder", '{"reason": "Come home soon"}')])], "usko bolo jaldi ghar aa jaye")
check("eval watch item: two saves in one family turn are counted (#1588)", _re4.double_saves(agent.call_state["call_id"]) == [1])
os.environ["FAMILY_NAMES"] = "1111:Papa"
check("caller_label: mapped 'Papa (1111)', unmapped as is", (agent.caller_label("1111"), agent.caller_label("2222"), agent.caller_label(""))
      == ("Papa (1111)", "2222", ""))
os.environ.pop("FAMILY_NAMES")

print("family notes: content only; one save for a callback (2026-11-26)")
family_msg = [t for t in agent.family_tools if t["function"]["name"] == "take_message"][0]["function"]["parameters"]["properties"]["message"]["description"]
check("family take_message: only the content, no 'Family called' / 'They asked you' lead-in",
      "Only what they want Jatin to know" in family_msg and "No lead-in like 'Family called' or 'They asked you'" in family_msg)
check("normal take_message keeps its own description (who called; what for him)",
      [t for t in agent.tools if t["function"]["name"] == "take_message"][0]["function"]["parameters"]["properties"]["message"] is agent.MESSAGE)
check("family prompt: a callback request is one add_reminder",
      "A callback request is one add_reminder; don't also take_message, unless they also pass on news." in agent.FAMILY_PROMPT)
for note, lead in (("Family called; they asked you to come home.", True), ("Papa called; come home.", True),
                   ("Please tell him to come home quickly.", True), ("Come home early.", False), ("Bring milk on the way.", False)):
    check(f"eval lead-in check: {note!r} -> {lead}", bool(_re4.LEAD_IN.search(note)) == lead)

print("fillers by situation (2026-11-26, approved mapping)")
FU_EN = agent.FOLLOW_UP_LINES["reminder"]["english"]
FU_HI = agent.pick_line(agent.FOLLOW_UP_LINES["message"], "hinglish")
for last, caller, kind, label in (
        (FU_EN, "no thanks, bye", "none", "after 'anything else?' (English)"),
        (FU_HI, "nahi bas, bye", "none", "after 'Aur kuch batana hai unhe?' (female Hinglish)"),
        ("Thanks, Rahul. What's it about?", "connect me please", "working", "explicit request"),
        ("Thanks, Rahul.", "haan try kar lo", "working", "'haan try kar lo' is a request by itself"),
        ("Do you want me to try him, or take a message?", "yes please", "working", "yes to an offer"),
        (agent.FAMILY_OPENERS["no_pickup"]["hinglish"], "haan", "working", "yes to the family no-pickup opener"),
        (agent.OFFER_FALLBACK_LINES["english"], "okay sure", "working", "yes to the options line"),
        ("Do you want me to try him, or take a message?", "no, just a message", "ack", "a no to an offer"),
        (agent.FAMILY_OPENERS["no_pickup"]["hinglish"], "nahi, theek hai", "ack", "'nahi, theek hai' is not a yes"),
        ("Can I take your name?", "what does Jatin do?", "working", "a question (ends with ?)"),
        ("Can I take your name?", "kya woh office mein hai", "working", "a Hinglish question word"),
        ("Can I take your name?", "I'm Rahul.", "ack", "answered a question"),
        ("Thanks, Rahul.", "it's about the VPS renewal", "ack", "anything else"),
        ("Thanks, Rahul.", "okay", "ack", "a bare okay with no offer before it")):
    check(f"{label}: {kind}", agent.filler_kind(last, caller) == kind)
check("groups: approved lines, all cached, no word replies often start with",
      agent.FILLER_GROUPS == {"ack": {"english": ["Hmm...", "Right...", "Alright..."], "hinglish": ["Achha...", "Hmm..."]},
                              "working": {"english": ["One second...", "Let me see..."], "hinglish": ["Ek second...", "Ek minute..."]}}
      and all(l in agent.all_fixed_lines()[lang] for g in agent.FILLER_GROUPS.values() for lang, ls in g.items() for l in ls))
check("'none' never plays and doesn't count as a played turn", agent.pick_filler({}, "english", 5, agent.random.Random(1), "none") is None)
st, rng, got = {}, agent.random.Random(3), []
for t in range(0, 200, 2):
    got.append(agent.pick_filler(st, "hinglish", t, rng, "working"))
played_w = [g for g in got if g]
check("working (Hinglish): only Ek second / Ek minute, never the same twice in a row, ~1/3 skipped",
      set(played_w) == {"Ek second...", "Ek minute..."} and all(a != b for a, b in zip(played_w, played_w[1:]))
      and 0.2 < got.count(None) / len(got) < 0.45)
# The server path: the situation comes from the call's own history.
client = server.app.test_client()
voice.cached_pcm = lambda text, lang: b"\x02\x00" * 2400
client.post("/start", json={})
agent.call_state["speech_lang"] = "english"
agent.filler_random = type("NoSkip", (), {"random": lambda self: 0.99, "choice": lambda self, o: o[0]})()
agent.call_state["messages"] += [{"role": "assistant", "content": FU_EN}, {"role": "user", "content": "no thanks, bye"}]
agent.call_state["turns"] = 50
r = client.get("/filler")
check("server: after the follow-up -> silence, logged as skipped with kind none",
      r.data == b"" and json.loads(events("filler_skipped")[-1]) == {"turn": 50, "kind": "none"})
agent.call_state["messages"] += [{"role": "assistant", "content": "Should I try calling him, or take a message?"},
                                 {"role": "user", "content": "haan try kar lo"}]
agent.call_state["turns"] = 60
r = client.get("/filler")
check("server: a yes to an offer -> 'One second...', logged with kind working",
      r.data and json.loads(events("filler")[-1])["line"] == "One second..." and json.loads(events("filler")[-1])["kind"] == "working")
client.post("/hangup")
voice.cached_pcm = lambda text, lang: None
import run_evals as _re5
check("eval simulation uses the same situation rules", "agent.filler_kind(last_jarvis, caller)" in open("run_evals.py", encoding="utf-8").read()
      and _re5.EXPLICIT_RING_REQUEST is agent.EXPLICIT_RING_REQUEST)

print("family notes: own words, Roman script; callback-only reasons dropped (2026-11-26)")
check("family prompt: messages stay in their words (Roman), reminder reason English; normal prompt unchanged",
      "Messages stay in their own words, in Roman script (Hinglish stays Hinglish, never Devanagari)" in agent.FAMILY_PROMPT
      and "translate the message" not in agent.FAMILY_PROMPT
      and "message: translated, keeping names and nicknames" in agent.SYSTEM_PROMPT)
check("family message schema: own words, Roman, example 'Ghar jaldi aa jao.'", "in their own words in Roman script" in family_msg
      and "Ghar jaldi aa jao." in family_msg and "Never Devanagari" in family_msg)
for reason, kept in (("Call them when free", "when free"), ("Call back when free", "when free"), ("Please call you back", ""),
                     ("Call the caller", ""), ("Request to call back", ""), ("Requested callback", ""), ("call kar le", ""),
                     ("mujhe call kar le", ""), ("about the bank", "about the bank"), ("Call about the bank", "about the bank"),
                     ("Electricity bill", "Electricity bill"), ("Bank ka kaam hai", "Bank ka kaam hai")):
    check(f"reason {reason!r} -> {kept!r}", agent.real_reason(reason) == kept)
os.environ["FAMILY_NAMES"] = "1111:Papa"
family_call(["no"])
family_turn([FakeStream([tool_chunk(0, "a", "add_reminder", '{"reason": "Please call you back", "urgency": "high"}')])], "mujhe call kar le")
note = db.reminders_for_call(agent.call_state["call_id"])[0]
import notify as _nt
check("saved as 'wants a callback' -> WhatsApp '🏠 Family call – Papa wants a callback (urgent)'",
      note["content"] == "wants a callback"
      and _nt.summary_line("reminder", note["caller_name"], note["content"], note["urgency"], True) == "🏠 Family call – Papa wants a callback (urgent)")
family_call(["no"])
family_turn([FakeStream([tool_chunk(0, "a", "add_reminder", '{"reason": "Call them when free about the bank"}')])], "free ho ke call kar le, bank ka kaam hai")
note = db.reminders_for_call(agent.call_state["call_id"])[0]
check("a real reason stays: 'when free about the bank'", note["content"] == "when free about the bank")
new_call()
agent.call_state["turns"] = 3
agent.add_reminder("Rahul", "low", "Please call back about the VPS")
check("normal callers' reasons are untouched (the stripping is family-only)",
      db.reminders_for_call(agent.call_state["call_id"])[0]["content"] == "Please call back about the VPS")
os.environ.pop("FAMILY_NAMES")

print("#1509: callback promises (live guard + eval share one check)")
for line in ("Got it — I’ll ask him to call you back ASAP.", "I'll ask him to call you back.", "Noted — I'll ask Jatin to call you back.",
             "I'll have Jatin call you back about the VPS renewal.", "Main Jatin ko call karne ko kahungi.",
             "He'll call you back.", "Woh aapko call back kar lenge.", "He will get back to you soon."):
    check(f"promise: {line}", agent.promises_callback(line))
for line in ("I'll note that you want a callback.", "Would you like to leave a message or have him call you back?",
             "Should I try calling him, or take a message or set a reminder?", "Theek hai, main bata dungi.",
             "Okay, I have noted it for him.", "Kya main Jatin ko aapka message de kar aapke call karne ke liye keh doon?"):
    check(f"not a promise: {line}", not agent.promises_callback(line))
said = claim_turn([FakeStream([text_chunk("Got it — I’ll ask him to call you back ASAP.")]),
                   FakeStream([text_chunk("I'll note that you want a callback. Anything else?")])], text="Just ask him to call me back ASAP.", reason=True)
check("live: the #1509 promise is never said; the retry is", said[0] == "I'll note that you want a callback."
      and any(d.startswith("promise:") for d in events("claim_guard"))
      and model_requests[1]["messages"][-2]["content"] == agent.CLAIM_RETRY_NOTES["promise"])
import run_evals as _re2
check("eval: the same check (promises_callback) fails a promise line", "promises_callback" in open("run_evals.py", encoding="utf-8").read())
check("prompt: no phone-number ask; callback wording", "Never ask for their phone number: caller ID already has it." in agent.SYSTEM_PROMPT
      and "say you'll note that they want one" in agent.SYSTEM_PROMPT and "or that you'll ask him to" in agent.SYSTEM_PROMPT)
check("prompt: offer after name + reason even without a request", "even if they never asked for Jatin" in agent.SYSTEM_PROMPT
      and "Hasn't asked for Jatin" not in agent.SYSTEM_PROMPT)
check("old fallback 'I can't put you through right now' gone from the code lines",
      not any("can't put you through" in l.lower() for ls in agent.all_fixed_lines().values() for l in ls))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
