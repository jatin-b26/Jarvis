# Zero-token check of server.py: Flask's test client plus a fake model. No API calls.
# Run from the project folder:   python test_server.py
import os
import sys
import json
import tempfile
from types import SimpleNamespace as N

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.DB_FILE = os.path.join(tempfile.gettempdir(), "jarvis_server_test.db")
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

# Fake text-to-speech: no Sarvam calls. Records which language each line was spoken in.
spoken_langs = []
def fake_speak(text, lang="english"):
    spoken_langs.append(lang)
    return b"fake-mp3", voice.VOICES.get(lang, voice.VOICES["english"])[0], 5, 10
real_speak, real_close_tts = voice.speak, voice.close_tts  # kept for the voice-set fallback test
voice.speak = fake_speak
voice.close_tts = lambda: None

# Step A: replies stream from /speech. Fake ElevenLabs stream: records the voice language of every
# streamed line (None = before the lock, i.e. Sia 06). Nothing is cached in these tests unless a test says so.
streamed = []
def fake_open_stream(text, voice_lang):
    streamed.append((text, voice_lang))
    return iter([b"\x01\x00" * 480]), voice.stream_voice(voice_lang)["name"], 5
real_open_stream, real_cached_pcm = voice.open_stream, voice.cached_pcm
voice.open_stream = fake_open_stream
voice.cached_pcm = lambda text, voice_lang: None
warmed = []
voice.warm_tts = lambda: warmed.append(1)   # no real ElevenLabs connection from tests


def hear(data):
    """Play a reply's audio the way the page does: GET its /speech URL."""
    return client.get(data["audio_url"]) if data.get("audio_url") else None

client = server.app.test_client()
agent.RING_SECONDS = 1  # unanswered rings time out fast in tests


def reply(content=None, tool_calls=None):
    msg = N(role="assistant", content=content, tool_calls=tool_calls)
    return N(choices=[N(message=msg)], usage=N(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None))


def judge():
    return reply(json.dumps({"language": "english", "caller_agreed_to_connect": False, "reason_known": True}))


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


def scripted(responses):
    queue = list(responses)
    return lambda **kw: take(queue, "response_format" in kw)


results = []
def check(name, condition):
    results.append(bool(condition))
    print(("  ok   " if condition else "  FAIL ") + name)


print("no hidden retries")
check("model client has SDK retries off (call_model's loop is the only retry logic)", agent.client.max_retries == 0)
check("transcription client has SDK retries off", voice.client().max_retries == 0)

print("server.py")
check("GET / serves the page", client.get("/").status_code == 200)
check("turn before start -> 400", client.post("/turn", json={"text": "hi"}).status_code == 400)

r = client.post("/start", json={}).get_json()
check("start -> opener, not ended", r["reply"] == agent.OPENER and r["ended"] is False)
check("empty text -> 400", client.post("/turn", json={"text": "  "}).status_code == 400)

agent.client.chat.completions.create = scripted([judge(), reply("Hi Priya, what's it about?")])
r = client.post("/turn", json={"text": "hi, this is priya"}).get_json()
check("turn -> Jarvis reply", r["reply"] == "Hi Priya, what's it about?" and r["ended"] is False)

bye = N(id="t1", type="function", function=N(name="end_call", arguments=json.dumps({"reason": "caller said bye", "kind": "done"})))
agent.client.chat.completions.create = scripted([judge(), reply(tool_calls=[bye]), reply("Bye!")])
r = client.post("/turn", json={"text": "bye"}).get_json()
check("end_call -> ended with outcome and call id", r["ended"] is True and r["outcome"] and r["call_id"])
check("turn after the call ended -> 400", client.post("/turn", json={"text": "hello?"}).status_code == 400)

client.post("/start", json={})
r = client.post("/hangup").get_json()
check("hang up -> ended, outcome abandoned", r["ended"] is True and r["outcome"] == "abandoned")

print("unexpected bug mid-call")
client.post("/start", json={})
real_run_agent = agent.run_agent
def boom(messages):
    raise KeyError("simulated bug")
agent.run_agent = boom
agent.client.chat.completions.create = scripted([judge()])
r = client.post("/turn", json={"text": "hi this is rahul"}).get_json()
agent.run_agent = real_run_agent
call_id = r["call_id"]
kinds = [row["kind"] for row in db.replay(call_id)]
check("no 500: caller hears the fallback line", r["reply"].startswith("Sorry, something's gone wrong on my end"))
check("call ended with outcome server_error", r["ended"] is True and r["outcome"] == "server_error")
check("server_error event + failed_call note saved",
      "server_error" in kinds and any(x["kind"] == "failed_call" for x in db.reminders_for_call(call_id)))

print("spoken turns (fake transcriber)")
import io
import time
import voice
from urllib.parse import unquote
real_transcribe = voice.transcribe
clip = lambda: {"audio": (io.BytesIO(b"fake-webm-bytes"), "turn.webm", "audio/webm")}

check("audio before start -> 400", client.post("/turn_audio", data=clip()).status_code == 400)
client.post("/start", json={})
check("no audio field -> 400", client.post("/turn_audio", data={}).status_code == 400)

def fake_stt(text, ms, engine="sarvam", why=None):
    """A fake voice.transcribe that also fills voice.last_stt the way the real one does."""
    def run(data, name, mime, seconds=None):
        voice.last_stt.update(engine=engine if text is not None else None, why=why)
        return text, ms
    return run


voice.transcribe = fake_stt("hi, this is priya", 120)
agent.client.chat.completions.create = scripted([judge(), reply("Hi Priya, what's it about?")])
r = client.post("/turn_audio", data=clip()).get_json()
check("clip -> heard text + Jarvis reply", r["heard"] == "hi, this is priya" and r["reply"] == "Hi Priya, what's it about?")
call_id = agent.call_state["call_id"]
check("stt line names the engine and its time", any(row["kind"] == "stt" and row["detail"] == "sarvam 120ms [clip 0 KB]" for row in db.replay(call_id)))
timings = [json.loads(row["detail"]) for row in db.replay(call_id) if row["kind"] == "timing"]
check("one timing line for the spoken turn, with the agent stages and stt",
      len(timings) == 1 and all(k in timings[0] for k in ("stt", "judge", "model", "tools", "ring"))
      and timings[0]["stt"] == 120 and len(timings[0]["model"]) == 1)
hear(r)
speech = [json.loads(row["detail"]) for row in db.replay(call_id) if row["kind"] == "speech"]
check("playing the reply logs a speech event: first audio + caller waited",
      len(speech) == 1 and speech[0]["first"] == 5 and speech[0]["waited"] >= 0 and speech[0]["cached"] is False)

for label, fake in [("transcription failed", fake_stt(None, 50, why="Sarvam failed: x; Scribe failed: y")),
                    ("heard nothing", fake_stt("", 50))]:
    voice.transcribe = fake
    turns_before = agent.call_state["turns"]
    r = client.post("/turn_audio", data=clip()).get_json()
    check(f"{label} -> 'didn't catch that', call continues, no agent turn",
          r["reply"] == server.DIDNT_CATCH and r["ended"] is False and agent.call_state["turns"] == turns_before
          and agent.call_state["messages"][-1]["content"] == server.DIDNT_CATCH)
check("both engines failed -> logged with the reasons",
      any(row["kind"] == "stt" and row["detail"].startswith("none 50ms (Sarvam failed: x; Scribe failed: y) [clip 0 KB], FAILED")
          for row in db.replay(agent.call_state["call_id"])))
# Scribe's Devanagari (fallback) reaches the model as is, and the page shows it as is: no display rewrite.
voice.transcribe = fake_stt("मेरा नाम सुरेश है", 300, engine="scribe", why="Sarvam failed: Sarvam 503")
agent.client.chat.completions.create = scripted([judge(), reply("Thanks, Suresh. Kis baare mein?")])
r = client.post("/turn_audio", data=clip()).get_json()
user_lines = [m["content"] for m in agent.call_state["messages"] if isinstance(m, dict) and m.get("role") == "user"]
check("Devanagari transcript -> the model and the page get it as is", user_lines[-1] == "मेरा नाम सुरेश है" and r["heard"] == "मेरा नाम सुरेश है")
check("... treated as Hinglish (Devanagari/Urdu script rule)", agent.call_state["lang"] == "hinglish")
check("... fallback logged with its reason", any(row["kind"] == "stt" and row["detail"] == "scribe 300ms (Sarvam failed: Sarvam 503) [clip 0 KB]"
                                                  for row in db.replay(agent.call_state["call_id"])))
check("no display rewrite and no Urdu retry left", not hasattr(voice, "roman_for_display") and not hasattr(server, "roman_pool")
      and "retried" not in voice.last_stt)
voice.transcribe = real_transcribe
client.post("/hangup")

print("speech-to-text: Sarvam Roman first, Scribe as the fallback (fake engines)")
used = []
real_sarvam, real_scribe = voice._sarvam, voice._scribe


def engines(sarvam, scribe):
    def fake(result, label):
        def run(data, name, mime):
            used.append(label)
            if isinstance(result, Exception):
                raise result
            return result
        return run
    voice._sarvam, voice._scribe = fake(sarvam, "sarvam"), fake(scribe, "scribe")
    used.clear()


engines("Haan ji, main Rohit bol raha hoon", "हाँ जी")
text, _ = voice.transcribe(b"x", "t.webm", "audio/webm", seconds=4.0)
check("normal turn: Sarvam only, Roman text", used == ["sarvam"] and text == "Haan ji, main Rohit bol raha hoon"
      and voice.last_stt == {"engine": "sarvam", "why": None})
engines(RuntimeError("Sarvam 500"), "हाँ जी")
text, _ = voice.transcribe(b"x", "t.webm", "audio/webm", seconds=4.0)
check("Sarvam fails -> Scribe", used == ["sarvam", "scribe"] and text == "हाँ जी"
      and voice.last_stt["engine"] == "scribe" and "Sarvam 500" in voice.last_stt["why"])
engines("should not be used", "a long message")
text, _ = voice.transcribe(b"x", "t.webm", "audio/webm", seconds=31.5)
check("over 30 s -> straight to Scribe, Sarvam never called", used == ["scribe"] and text == "a long message"
      and voice.last_stt["why"] == "over 30 s")
engines("short", "x")
voice.transcribe(b"x", "t.webm", "audio/webm", seconds=None)
check("length unknown -> Sarvam (a too-long clip then fails over to Scribe)", used == ["sarvam"])
engines(RuntimeError("Sarvam 400"), RuntimeError("Scribe 401"))
text, _ = voice.transcribe(b"x", "t.webm", "audio/webm", seconds=3)
check("both fail -> None, never raises", text is None and voice.last_stt["engine"] is None and "Scribe 401" in voice.last_stt["why"])
engines("", "x")
text, _ = voice.transcribe(b"x", "t.webm", "audio/webm", seconds=1)
check("Sarvam hears nothing -> empty text, no second engine (silence isn't a failure)", text == "" and used == ["sarvam"])
voice._sarvam, voice._scribe = real_sarvam, real_scribe
import inspect as _insp
_src = _insp.getsource(voice._sarvam) + _insp.getsource(voice._scribe)
check("request settings: saaras:v4 translit; scribe_v3 Hindi + English; no prompt or hint words",
      '"saaras:v4"' in _src and '"translit"' in _src and '"scribe_v3"' in _src and '"hin"' in _src
      and "prompt" not in _src and not hasattr(voice, "STT_PROMPT"))
captured = {}
real_t = voice.transcribe
voice.transcribe = lambda data, name, mime, seconds=None: (captured.update(seconds=seconds) or ("ok then", 10))
client.post("/start", json={})
agent.client.chat.completions.create = scripted([judge(), reply("Sure.")])
client.post("/turn_audio", data={**clip(), "seconds": "31.20"})
check("the page's clip length reaches transcribe", captured.get("seconds") == 31.2)
voice.transcribe = real_t
client.post("/hangup")

for _text, _want in [("", True), ("...", True), ("Hmm.", True), ("Umm, uh", True), ("हम्म", True),
                     ("Jarvis, Jatin, phone call, English, Hindi, Hinglish", True),
                     ("haan", False), ("Haan.", False), ("ok", False), ("bye", False), ("Thank you.", False),
                     ("Jatin se baat karni thi", False), ("hmm, ok try him", False)]:
    check(f"not_speech({_text!r}) == {_want}", voice.not_speech(_text) == _want)
voice.transcribe = fake_stt("Hmm.", 60)
client.post("/start", json={})
_turns = agent.call_state["turns"]
_r = client.post("/turn_audio", data=clip()).get_json()
check("filler-only transcript -> 'didn't catch that', the model never sees it",
      _r["reply"] == server.DIDNT_CATCH and agent.call_state["turns"] == _turns
      and any(row["detail"].endswith("not speech: 'Hmm.'") for row in db.replay(agent.call_state["call_id"]) if row["kind"] == "stt"))
voice.transcribe = real_transcribe

print("prompt rules from the live test")
for name, prompt in (("SYSTEM", agent.SYSTEM_PROMPT), ("FAMILY", agent.FAMILY_PROMPT)):
    check(f"{name}: one general honesty rule (old line folded in)",
          "Only say what you know from these instructions, this call or a tool." in prompt and "don't explain, don't guess" not in prompt)
    check(f"{name}: callback steer removed from the offer rule" if name == "SYSTEM" else f"{name}: (no offer rule)",
          "save a reminder instead;" in prompt if name == "SYSTEM" else True)
    check(f"{name}: off-topic rule: one kind line, no opinion or story (persona, 2026-11-25)",
          "no opinion or story" in prompt and "line at most" not in prompt)
    check(f"{name}: second-save confirmation is said by code; no 'and wait' left",
          "the confirmation is said for you" in prompt and "and wait" not in prompt)
    check(f"{name}: no life of her own; no 'friend' anywhere",
          "You have no life of your own: no opinions, no experiences" in prompt
          and prompt.lower().count("friend") == (1 if name == "SYSTEM" else 0))  # only "like friends" in the Hinglish line

print("text-to-speech: streamed PCM, voice lock (fake ElevenLabs)")
r = client.post("/start", json={}).get_json()
check("opener comes with an audio URL, not inline audio", r["audio_url"].startswith("speech/") and "audio" not in r)
streamed.clear()
a = hear(r)
check("GET /speech streams PCM with its sample rate", a.status_code == 200 and a.mimetype == "audio/pcm"
      and a.headers["X-Sample-Rate"] == "24000" and len(a.data) == 960)
check("opener: Sia 05, spoken as Hinglish (no accent tag)", streamed == [(agent.OPENER, "hinglish")]
      and voice.stream_voice("hinglish")["name"] == "sia05" and voice.stream_voice("hinglish")["prefix"] == "")
check("a line plays once (token used up)", client.get(r["audio_url"]).status_code == 404)

# One voice (Sia 05) for both languages; how a line is SAID follows the caller's latest substantive turn.
agent.client.chat.completions.create = scripted([
    reply(json.dumps({"language": "hinglish", "caller_agreed_to_connect": False, "reason_known": False})),
    reply("Haan ji, aap kaun bol rahe hain?")])
streamed.clear()
r = client.post("/turn", json={"text": "jatin hai kya"}).get_json()
hear(r)
call_id = agent.call_state["call_id"]
check("clear Hinglish turn -> spoken as Hinglish, no tag", streamed == [(r["reply"], "hinglish")])
agent.client.chat.completions.create = scripted([judge(), reply("Got it. What's it about?")])  # judge now says english
streamed.clear()
hear(client.post("/turn", json={"text": "my name is Rohit"}).get_json())
check("caller switches to English -> spoken as English (with the accent tag)", [v for _, v in streamed] == ["english"]
      and voice.stream_voice("english")["prefix"] == "[Indian English accent] ")
check("every voice config is Sia 05 (one voice, Sia 06 gone)",
      {v["id"] for v in voice.SIA.values()} == {voice.SIA05} and not hasattr(voice, "SIA06"))
check("no voice lock any more", not any(row["kind"] == "voice_locked" for row in db.replay(call_id)))
client.post("/hangup")

# Very short turns keep the previous language (#242): "bye" after Hinglish stays Hinglish.
client.post("/start", json={})
agent.client.chat.completions.create = scripted([
    reply(json.dumps({"language": "hinglish", "caller_agreed_to_connect": False, "reason_known": True})),
    reply("Theek hai, kya message dena hai?")])
client.post("/turn", json={"text": "mera naam Rohit hai, site ke baare mein"})
agent.client.chat.completions.create = scripted([judge(), reply("Okay.")])  # judge reads "bye" as English
client.post("/turn", json={"text": "ok bye"})
check("'ok bye' after a Hinglish turn -> fixed lines stay Hinglish (#242)", agent.fixed_line_lang() == "hinglish")
client.post("/hangup")

# A bare "hello" doesn't set the language either.
client.post("/start", json={})
agent.client.chat.completions.create = scripted([judge(), reply("Hi! Can I take your name?")])
client.post("/turn", json={"text": "hello"})
check("'hello' -> no substantive language yet", agent.call_state["speech_lang"] is None)
agent.client.chat.completions.create = scripted([judge(), reply("Thanks, Priya. What's it about?")])
client.post("/turn", json={"text": "hi, this is priya calling"})
check("first clear English turn -> English", agent.call_state["speech_lang"] == "english")
check("language_is_clear: greetings and 1-2 words are not clear",
      not any(agent.language_is_clear(t) for t in ["hello", "hi there", "haan ji", "ok sure", "namaste ji"])
      and all(agent.language_is_clear(t) for t in ["hi this is priya", "jatin hai kya", "mera naam Rohit hai"]))
client.post("/hangup")

# Cached fixed lines never touch ElevenLabs.
r = client.post("/start", json={}).get_json()
voice.cached_pcm = lambda text, voice_lang: b"\x02\x00" * 100 if text == agent.OPENER else None
streamed.clear()
a = hear(r)
speech = [json.loads(row["detail"]) for row in db.replay(agent.call_state["call_id"]) if row["kind"] == "speech"]
check("cached opener -> served from the cache, no ElevenLabs request", streamed == [] and len(a.data) == 200
      and speech[-1]["cached"] is True)
voice.cached_pcm = lambda text, voice_lang: None
client.post("/hangup")

# ElevenLabs down (no backup voice): cached outage line in the call's voice, note saved, call ended.
client.post("/start", json={})
agent.client.chat.completions.create = scripted([judge(), reply("Thanks, Rohit. What's it about?")])
r = client.post("/turn", json={"text": "hi this is rohit calling"}).get_json()
call_id = agent.call_state["call_id"]
voice.open_stream = lambda text, voice_lang: None
outage_line = agent.LINES_NOTE_SAVED["english"]
voice.cached_pcm = lambda text, voice_lang: b"\x03\x00" * 50 if (text, voice_lang) == (outage_line, "english") else None
a = hear(r)
check("TTS down -> the response says the call ended, with the outage line as text",
      a.headers.get("X-Call-Ended") == "1" and unquote(a.headers["X-Reply"]) == outage_line and len(a.data) == 100)
check("TTS down -> outcome tts_unavailable, failed_call note saved, call over",
      a.headers["X-Outcome"] == "tts_unavailable" and server.active["call"] is False
      and any(x["kind"] == "failed_call" for x in db.reminders_for_call(call_id)))
check("TTS down -> logged (tts_unavailable + failed speech event)",
      any(row["kind"] == "tts_unavailable" for row in db.replay(call_id))
      and any(row["kind"] == "speech" and json.loads(row["detail"]).get("failed") for row in db.replay(call_id)))
voice.open_stream = fake_open_stream
voice.cached_pcm = lambda text, voice_lang: None

print("judge facts: reason_known accumulates, caller_agreed is live")
client.post("/start", json={})
agent.call_state.update(wants_jatin_turn=0)
verdict = lambda agreed, known: reply(json.dumps({"language": "english", "caller_agreed_to_connect": agreed, "reason_known": known}))
agent.client.chat.completions.create = scripted([verdict(True, True), reply("Okay.")])
client.post("/turn", json={"text": "it's about the invoice, yes try him"})
agent.client.chat.completions.create = scripted([verdict(False, False), reply("Okay.")])  # judge flips both off
client.post("/turn", json={"text": "actually no, don't"})
check("reason_known stays True once known (judge can't switch it off)", agent.call_state["reason_known"] is True)
check("caller_agreed follows the latest turn (caller changed their mind)", agent.call_state["caller_agreed"] is False)
client.post("/hangup")

print("judge runs alongside the model (Week 3)")
import time as _time


def slow(fake, judge_s=0.0, model_s=0.0):
    """Wrap a fake so judge requests and model requests take this long."""
    def wrapped(**kw):
        _time.sleep(judge_s if "response_format" in kw else model_s)
        return fake(**kw)
    return wrapped


# 1. Overlap: judge 400 ms and model 400 ms should cost ~400 ms, not ~800.
client.post("/start", json={})
agent.client.chat.completions.create = slow(scripted([judge(), reply("Hi Priya, what's it about?")]), 0.4, 0.4)
began = _time.time()
client.post("/turn", json={"text": "hi this is priya"})
took = _time.time() - began
t = [json.loads(row["detail"]) for row in db.replay(agent.call_state["call_id"]) if row["kind"] == "timing"][-1]
check(f"judge and model overlap: turn took {took:.2f}s for 0.4s + 0.4s of work", took < 0.7)
check("timing records judge time and how long the turn waited for it", t["judge"] >= 350 and t["judge_wait"] < 200)
client.post("/hangup")


# 2. The ring lock waits for THIS turn's verdict, even when the model gets there first.
def ring_attempt(agreed):
    """Turn 1: caller asks for Jatin. Turn 2: the model calls try_reach_owner at once, while a slow judge
    (300 ms) is still deciding whether the caller agreed. Returns the call summary."""
    ring = N(id="t9", type="function", function=N(name="try_reach_owner", arguments=json.dumps(
        {"caller_name": "Priya", "reason": "invoice for the website work"})))
    wants = N(id="t8", type="function", function=N(name="caller_wants_jatin", arguments="{}"))
    v1 = reply(json.dumps({"language": "english", "caller_agreed_to_connect": False, "reason_known": True}))
    v2 = reply(json.dumps({"language": "english", "caller_agreed_to_connect": agreed, "reason_known": True}))
    agent.client.chat.completions.create = slow(scripted([
        v1, reply(tool_calls=[wants]), reply("Want me to try him?"),
        v2, reply(tool_calls=[ring]), reply("Okay.")]), judge_s=0.3)
    return agent.run_call(caller_lines=["priya here, invoice for the website, can i talk to jatin", "yes try him"],
                          owner_answers=["yes"], verbose=False)


r = ring_attempt(agreed=True)
check("slow judge says agreed -> ring goes through and connects", r["connected"] is True)
r = ring_attempt(agreed=False)
check("slow judge says NOT agreed -> lock blocks the ring (no stale verdict)",
      r["connected"] is False and "try_reach_owner" in r["tools_blocked"])

print("talk: short presses ignored, hold Space to talk (2026-11-26)")
_pg = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"), encoding="utf-8").read()
check("short press: MIN_PRESS_MS 300, clip dropped silently, barge-in waits for a real press",
      "const MIN_PRESS_MS = 300;" in _pg and "if (discardClip) { discardClip = false; chunks = []; return; }" in _pg
      and "bargeTimer = setTimeout(() => { bargeTimer = null; bargeIn(); }, MIN_PRESS_MS)" in _pg
      and "clearTimeout(bargeTimer);" in _pg)
check("Space: keydown starts, keyup stops, auto-repeat ignored, not while typing in a field",
      'event.code !== "Space"' in _pg and "if (event.repeat) return;" in _pg and "typingInField(event.target)" in _pg
      and '["INPUT", "TEXTAREA", "SELECT"]' in _pg and "startTalking();" in _pg and "stopTalking();" in _pg)
check("same path as the button: the 0.4 s tail is shared (TAIL_MS 400)", "const TAIL_MS = 400;" in _pg)
check("hint on the page", "Hold Space or the button to talk." in _pg)

print("caller IDs on the page: family names from the server at runtime, never in the page file")
_names = os.environ.get("FAMILY_NAMES")
os.environ["FAMILY_NAMES"] = "1111:Papa,2222:Mummy"
ids = client.get("/caller_ids").get_json()["family"]
check("/caller_ids: 'Papa (1111)', 'Mummy (2222)', unmapped '3333' as is",
      {i["id"]: i["label"] for i in ids} == {"1111": "Papa (1111)", "2222": "Mummy (2222)", "3333": "3333"})
os.environ["FAMILY_NAMES"] = "1111:Dadi"
check("read at request time (a changed .env value shows without a page change)",
      client.get("/caller_ids").get_json()["family"][0]["label"] == "Dadi (1111)")
if _names is None:
    os.environ.pop("FAMILY_NAMES")
else:
    os.environ["FAMILY_NAMES"] = _names
from dotenv import dotenv_values
_page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"), encoding="utf-8").read()
_real = [part.split(":", 1)[1].strip() for part in (dotenv_values(".env").get("FAMILY_NAMES") or "").split(",") if ":" in part]
check(f"no FAMILY_NAMES name from .env is written into static/index.html ({len(_real)} checked)",
      not any(n and n in _page for n in _real) and 'fetch("caller_ids")' in _page and 'list="familyIds"' in _page)

print("family caller, Jatin doesn't pick up")
r = client.post("/start", json={"caller_id": "1111"}).get_json()
check("family opener after an unanswered ring = the no-pickup opener", r["reply"] == agent.FAMILY_OPENERS["no_pickup"]["hinglish"]
      and r["ended"] is False)
check("first family ring: no hold line (Jarvis hasn't spoken yet)",
      not any(row["detail"].startswith("(hold line)") for row in db.replay(r["call_id"]) if row["kind"] == "jarvis_said"))
_ring_s, agent.RING_SECONDS = agent.RING_SECONDS, 0.2
check("family re-ring (rings >= 1): the hold line is spoken, no answer -> ''",
      server.ring_owner("Family call again") == ""
      and any(row["detail"].startswith("(hold line)") for row in db.replay(r["call_id"]) if row["kind"] == "jarvis_said"))
agent.RING_SECONDS = _ring_s
client.post("/hangup")

print("Jatin's phone (step 5)")
import threading
import time
phone_client = server.app.test_client()  # a second "browser tab": the page polling /ring and clicking

check("/ring when nothing rings -> not ringing", phone_client.get("/ring").get_json()["ringing"] is False)
check("/answer when nothing rings -> 409", phone_client.post("/answer", json={"answer": "accept"}).status_code == 409)
check("/answer with a junk choice -> 400", phone_client.post("/answer", json={"answer": "maybe"}).status_code == 400)


def ring_turn(click):
    """Run one turn where the model rings Jatin. click: "accept", "decline" or None (let it time out).
    Returns (the turn's JSON, what /ring showed, whether the lock was held during the ring)."""
    client.post("/start", json={})
    agent.call_state.update(wants_jatin_turn=0)  # as if caller_wants_jatin ran on an earlier turn
    ring_call = N(id="t2", type="function", function=N(name="try_reach_owner", arguments=json.dumps(
        {"caller_name": "Priya", "reason": "invoice for the website work"})))
    agent.client.chat.completions.create = scripted([
        reply(json.dumps({"language": "english", "caller_agreed_to_connect": True, "reason_known": True})),
        reply(tool_calls=[ring_call]), reply("Okay, wrapping up.")])
    out = {}
    worker = threading.Thread(target=lambda: out.update(r=client.post("/turn", json={"text": "yes please"}).get_json()))
    worker.start()
    seen = {"ringing": False}
    for _ in range(100):  # wait up to 5 s for the ring to appear
        seen = phone_client.get("/ring").get_json()
        if seen["ringing"]:
            break
        time.sleep(0.05)
    locked = server.call_lock.locked()
    if click and seen["ringing"]:
        time.sleep(0.3)  # Jatin takes a moment to click, so the ring wait is measurable
        phone_client.post("/answer", json={"answer": click})
    worker.join(timeout=10)
    return out.get("r"), seen, locked


r, seen, locked = ring_turn("accept")
check("ring shows the briefing and the hold line audio",
      seen["ringing"] and "Priya" in seen["briefing"] and seen["hold_audio_url"].startswith("speech/"))
check("call lock is held during the ring (so /hangup waits)", locked)
check("Accept -> connected, call ended", r and r["ended"] is True and r["outcome"] == "connected")
kinds = [(row["kind"], row["detail"]) for row in db.replay(r["call_id"])]
check("ring + owner decision logged", ("ring", "accepted") in kinds and any(k == "owner_decision" and "ACCEPTED" in d for k, d in kinds))
check("hold line logged as said", any(k == "jarvis_said" and d.startswith("(hold line)") for k, d in kinds))
check("phone stops ringing afterwards", phone_client.get("/ring").get_json()["ringing"] is False)
t = [json.loads(row["detail"]) for row in db.replay(r["call_id"]) if row["kind"] == "timing"][-1]
check("ring wait timed on its own, not inside tools; accepted = one model request (code says 'connecting')",
      t["ring"] >= 250 and t["tools"] < 250 and len(t["model"]) == 1)

r, seen, locked = ring_turn("decline")
check("Decline -> call continues, not connected", r and r["ended"] is False and agent.call_state["connected"] is False)
check("decline logged", any(row["kind"] == "ring" and row["detail"] == "declined" for row in db.replay(agent.call_state["call_id"])))
client.post("/hangup")

r, seen, locked = ring_turn(None)
check("no click -> times out after RING_SECONDS, call continues", r and r["ended"] is False)
check("no answer logged", any(row["kind"] == "ring" and row["detail"] == "no answer in 1s"
                              for row in db.replay(agent.call_state["call_id"])))
client.post("/hangup")

print("family caller, Jatin accepts")
out = {}
worker = threading.Thread(target=lambda: out.update(r=client.post("/start", json={"caller_id": "1111"}).get_json()))
worker.start()
for _ in range(100):
    seen = phone_client.get("/ring").get_json()
    if seen["ringing"]:
        break
    time.sleep(0.05)
check("family ring has no hold line (Jarvis hasn't spoken yet)", seen["ringing"] and seen["hold_audio_url"] is None)
phone_client.post("/answer", json={"answer": "accept"})
worker.join(timeout=10)
check("family Accept -> connected straight away, no opener",
      out.get("r") and out["r"]["ended"] is True and out["r"]["outcome"] == "connected" and out["r"]["reply"] is None)

print("closing line: code says goodbye after end_call, not the model")
def end_turn(reason, lang="english", setup=None, text="bye", kind="done"):
    """One turn where the model calls end_call. Returns (reply, model requests made, events)."""
    client.post("/start", json={})
    if setup:
        setup()
    seen = []
    queue = [reply(json.dumps({"language": lang, "caller_agreed_to_connect": False, "reason_known": True})),
             reply(tool_calls=[N(id="e1", type="function", function=N(name="end_call", arguments=json.dumps({"reason": reason, "kind": kind})))]),
             reply("The assistant will end the call now. Bye.")]  # must never be asked for
    def fake(**kw):
        if "response_format" not in kw:
            seen.append(kw)
        return take(queue, "response_format" in kw)
    agent.client.chat.completions.create = fake
    r = client.post("/turn", json={"text": text}).get_json()
    return r, len(seen), [(row["kind"], row["detail"]) for row in db.replay(r["call_id"])]

r, model_calls, events = end_turn("caller said bye")
check("end_call -> exactly one model request (no goodbye turn)", model_calls == 1)
check("nothing saved -> fixed 'nothing_saved' line", r["reply"] == agent.CLOSING_LINES["nothing_saved"]["english"] and r["ended"])
check("closing line logged as said + fixed_line event", ("jarvis_said", r["reply"]) in events and ("fixed_line", "closing line: nothing_saved") in events)

r, _, _ = end_turn("sales call", kind="spam")
check("kind=spam -> flat spam line, outcome spam", r["reply"] == agent.CLOSING_LINES["spam"]["english"] and r["outcome"] == "spam")
r, _, _ = end_turn("abusive caller after one warning", kind="abuse")
check("kind=abuse -> flat abuse line", r["reply"] == agent.CLOSING_LINES["abuse"]["english"])
# Run B: the kind picks the line, not keywords in the reason (#322: a scam got the warm goodbye).
r, _, _ = end_turn("asked for an OTP, ended", kind="spam")
check("scam with no keyword in the reason, kind=spam -> spam line (#322)", r["reply"] == agent.CLOSING_LINES["spam"]["english"])
r, _, _ = end_turn("sales call", kind="done")
check("'sales' in the reason but kind=done -> no spam line (keyword match gone)",
      r["reply"] == agent.CLOSING_LINES["nothing_saved"]["english"] and r["outcome"] != "spam")
check("end_call schema requires kind with the three values",
      [t for t in agent.tools if t["function"]["name"] == "end_call"][0]["function"]["parameters"]["required"] == ["reason", "kind"]
      and list(agent.END_KINDS) == ["done", "spam", "abuse"])
r, _, _ = end_turn("caller said bye", setup=lambda: agent.call_state.update(outcome="reminder"))
check("reminder saved -> reminder line", r["reply"] == agent.CLOSING_LINES["reminder"]["english"])
r, _, _ = end_turn("caller said bye", setup=lambda: agent.call_state.update(outcome="message", speech_lang="hinglish"))
check("Hinglish call, judge reads final 'bye' as English -> still the Hinglish line (locked voice, #242)",
      r["reply"] == agent.pick_line(agent.CLOSING_LINES["message"], "hinglish"))
r, _, _ = end_turn("caller said bye", setup=lambda: agent.call_state.update(family=True, outcome="message"))
check("family call -> family line", r["reply"] == agent.CLOSING_LINES["family_message"]["english"])
check("every closing line passes leak_check against the live prompts",
      all(agent.leak_check(line, agent.state_hint(), prompt) is None
          for row in agent.CLOSING_LINES.values() for line in row.values()
          for prompt in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))

print("leak guard")
check("retry note passes leak_check against the live hint and both system prompts",
      all(agent.leak_check(agent.LEAK_RETRY_NOTE, agent.state_hint(), p) is None for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))
for known in ["The assistant must reply now in Hinglish. Say a quick friendly bye.",            # 216
              "(assistant couldn't reply in two voices; continuing...)",                        # 242
              "The assistant will end the call now. Bye.",                                      # 204
              "The conversation already ended. I must respond in the caller's language."]:      # 238
    check(f"self_reference catches: {known[:45]}", agent.leak_check(known, "", "") == "self_reference")
check("hint_overlap catches a prompt recitation",
      agent.leak_check("Wanting to talk to Jatin is not a reason, so tell me.", "", agent.SYSTEM_PROMPT) == "hint_overlap")
for clean in ["Hi, Jarvis here, Jatin's AI assistant. He can't pick up right now.",
              "Can I take your name, please?", "Kya aap apna naam bata sakte hain?", "I'm his assistant, he's not on this line.",
              # Jatin's style example (2026-11-23): saying it must never trip the guard, so the rule stays under 6 shared words
              "Kya aap apna naam aur aapko Jatin se kya kaam tha, woh bata sakte hain?",
              "Can you please tell me your name and what you wanted to talk to Jatin about?"]:
    check(f"clean line passes: {clean[:45]}", agent.leak_check(clean, agent.state_hint(), agent.SYSTEM_PROMPT) is None)

def leak_turn(*model_replies, connected=False):
    client.post("/start", json={})
    agent.call_state.update(connected=connected)
    sent = []
    queue = [judge()] + [reply(t) for t in model_replies]
    def fake(**kw):
        if "response_format" not in kw:
            sent.append(kw["messages"])
        return take(queue, "response_format" in kw)
    agent.client.chat.completions.create = fake
    r = client.post("/turn", json={"text": "hi this is priya"}).get_json()
    return r, sent, [(row["kind"], row["detail"]) for row in db.replay(agent.call_state["call_id"])]

r, sent, events = leak_turn("The assistant must reply now in English.", "Hi Priya, what did you want to talk to him about?")
check("leak -> retried, caller hears the clean retry", r["reply"] == "Hi Priya, what did you want to talk to him about?")
check("leaked line never logged as said", not any(k == "jarvis_said" and "The assistant" in d for k, d in events))
check("catch logged with the check name", any(k == "leak_guard" and d.startswith("self_reference: The assistant") for k, d in events))
check("retry request carries the retry note", len(sent) == 2 and any(m.get("content") == agent.LEAK_RETRY_NOTE for m in sent[1] if isinstance(m, dict)))

r, _, events = leak_turn("The assistant must reply now.", "The caller wants a reply.")
check("retry leaks too -> fixed fallback line", r["reply"] == agent.LEAK_FALLBACK_LINES["other"]["english"])
check("both catches + fallback logged", sum(k == "leak_guard" for k, _ in events) == 2 and ("fixed_line", "retry leaked too: other") in events)
r, _, _ = leak_turn("The assistant must reply now.", "The caller wants a reply.", connected=True)
check("double leak on a connected call -> 'connecting you' fallback", r["reply"] == agent.LEAK_FALLBACK_LINES["connected"]["english"])
client.post("/hangup")

print("step 4: code-spoken follow-up")
def tool(name, **args):
    return N(id=f"c{name}", type="function", function=N(name=name, arguments=json.dumps(args)))

def stubbed(model_replies, lang="english"):
    """Judge replies are generated on demand; model replies come from the list. Returns the list of
    model (non-judge) requests made, so a test can count them."""
    queue = list(model_replies)
    made = []
    def fake(**kw):
        if "response_format" in kw:
            return reply(json.dumps({"language": lang, "caller_agreed_to_connect": False, "reason_known": True}))
        made.append(kw)
        return queue.pop(0) if queue else reply("(model asked but the script is empty)")
    agent.client.chat.completions.create = fake
    return made

def said_and_fixed(call_id):
    rows = db.replay(call_id)
    return ([r["detail"] for r in rows if r["kind"] == "jarvis_said"],
            [r["detail"] for r in rows if r["kind"] == "fixed_line"])

REM = lambda: tool("add_reminder", caller_name="Rahul", reason="VPS renewal", urgency="low")
MSG = lambda: tool("take_message", caller_name="Rahul", message="He called about the VPS.")

# 1. Reminder: follow-up spoken by code, no model request for it, turn ends there.
client.post("/start", json={})
made = stubbed([reply(tool_calls=[REM()]), reply("Should never be asked for.")])
r = client.post("/turn", json={"text": "it's about the vps renewal, can he call me back?"}).get_json()
cid = agent.call_state["call_id"]
check("reminder -> code-spoken follow-up line", r["reply"] == agent.FOLLOW_UP_LINES["reminder"]["english"] and r["ended"] is False)
check("follow-up made no model request (one request: the save)", len(made) == 1)
check("follow-up logged as said + fixed_line event", r["reply"] in said_and_fixed(cid)[0] and "follow-up: reminder" in said_and_fixed(cid)[1])
# ... then "no" -> end_call -> neutral closing line (the follow-up already confirmed the save)
made = stubbed([reply(tool_calls=[tool("end_call", reason="caller said no, bye", kind="done")])])
r = client.post("/turn", json={"text": "no, bye"}).get_json()
check("after the follow-up, 'no' -> neutral closing line", r["reply"] == agent.CLOSING_LINES["after_follow_up"]["english"] and r["ended"])
check("outcome stays reminder", r["outcome"] == "reminder")

# 2. Text the model wrote alongside the save is dropped, never said.
client.post("/start", json={})
stubbed([reply("Noted! Anything else?", tool_calls=[MSG()])])
r = client.post("/turn", json={"text": "just tell him i called"}).get_json()
said, _ = said_and_fixed(agent.call_state["call_id"])
check("model text in a save turn is dropped; caller hears only the follow-up",
      r["reply"] == agent.FOLLOW_UP_LINES["message"]["english"] and "Noted! Anything else?" not in said)
check("dropped text logged for the replay",
      any(row["kind"] == "dropped_text" and "Noted! Anything else?" in row["detail"] for row in db.replay(agent.call_state["call_id"])))

# 3. They add something after the follow-up (Jatin's ruling (b), #381): saved, a short model confirmation,
#    no second follow-up, and the call WAITS for their bye like any other turn.
made = stubbed([reply(tool_calls=[tool("take_message", caller_name="Rahul", message="Also: the invoice is late.")]),
                reply("Okay, I'll tell him that too.")])
r = client.post("/turn", json={"text": "haan, also tell him the invoice is late"}).get_json()
said, fixed = said_and_fixed(agent.call_state["call_id"])
tool_results = [row["detail"] for row in db.replay(agent.call_state["call_id"]) if row["kind"] == "tool_result"]
check("second save -> result says the confirmation was heard, never end_call",
      tool_results[-1].endswith("They've heard it confirmed.") and not any("Now end_call" in d for d in tool_results))
check("second save -> code's confirmation is spoken (no model reply), call stays open",
      r["reply"] == agent.ADDED_LINES["english"] and r["ended"] is False and len(made) == 1)
check("follow-up spoken once per call even after a second save", sum(f.startswith("follow-up") for f in fixed) == 1)
check("both saves are in the database", len(db.reminders_for_call(agent.call_state["call_id"])) == 2)
# ... then they say bye -> end_call -> the neutral closing line
stubbed([reply(tool_calls=[tool("end_call", reason="caller said bye", kind="done")])])
r = client.post("/turn", json={"text": "bas, bye"}).get_json()
check("bye after the second save -> end_call -> neutral closing line",
      r["ended"] and r["reply"] == agent.CLOSING_LINES["after_follow_up"]["english"])
# Second save + end_call in one batch: end_call refused, the call waits.
client.post("/start", json={})
stubbed([reply(tool_calls=[MSG()])])
client.post("/turn", json={"text": "tell him i called"})
made = stubbed([reply(tool_calls=[tool("take_message", caller_name="Rahul", message="Also: the invoice is late."),
                                  tool("end_call", reason="caller added one thing", kind="done")]),
                reply("Should never be asked for.")])
r = client.post("/turn", json={"text": "also the invoice is late, bye"}).get_json()
check("second save + end_call in one batch -> allowed (only the first save blocks end_call), neutral line",
      r["ended"] and r["reply"] == agent.CLOSING_LINES["after_follow_up"]["english"]
      and "end_call" not in agent.call_state["tools_blocked"] and len(made) == 1)
client.post("/hangup")

# 4. The follow-up ALWAYS comes after the first save (caller_done was removed 2026-11-23: the model
#    guessed "done" from "just leave a message" and 14 of 21 save calls ended before the caller said bye).
check("caller_done is gone from every tool schema",
      "caller_done" not in json.dumps(agent.tools + agent.family_tools))
client.post("/start", json={})
made = stubbed([reply(tool_calls=[REM()]), reply(tool_calls=[tool("end_call", reason="caller said that's all, bye", kind="done")])])
r = client.post("/turn", json={"text": "vps renewal, that's all, bye"}).get_json()
check("first save, even with 'that's all, bye' in the line -> follow-up, call stays open",
      r["reply"] == agent.FOLLOW_UP_LINES["reminder"]["english"] and r["ended"] is False and len(made) == 1)
client.post("/start", json={})
stubbed([reply(tool_calls=[REM(), tool("end_call", reason="caller said bye", kind="done")])])
r = client.post("/turn", json={"text": "vps renewal, bye"}).get_json()
ev = [(row["kind"], row["detail"]) for row in db.replay(agent.call_state["call_id"])]
check("save + end_call in one batch -> end_call refused, follow-up said, call stays open",
      r["reply"] == agent.FOLLOW_UP_LINES["reminder"]["english"] and r["ended"] is False
      and any(k == "tool_result" and d.startswith("end_call -> ERROR: can't end the call in the same turn as the first save") for k, d in ev))
check("the refused end_call counts as a blocked attempt", "end_call" in agent.call_state["tools_blocked"])
client.post("/start", json={})
stubbed([reply(tool_calls=[tool("end_call", reason="caller said bye", kind="done"), MSG()])])
r = client.post("/turn", json={"text": "tell him i called, bye"}).get_json()
check("end_call BEFORE the save in the batch -> still refused, follow-up said",
      r["reply"] == agent.FOLLOW_UP_LINES["message"]["english"] and r["ended"] is False)
client.post("/hangup")

# Run B: an end_call with a kind outside done/spam/abuse is refused and the call goes on.
client.post("/start", json={})
stubbed([reply(tool_calls=[tool("end_call", reason="caller said bye", kind="maybe")]), reply("Okay, anything else?")])
r = client.post("/turn", json={"text": "bye"}).get_json()
check("bad kind -> refused, call not ended",
      r["ended"] is False and "end_call" in agent.call_state["tools_blocked"]
      and any(row["kind"] == "tool_result" and "kind must be done, spam or abuse" in row["detail"] for row in db.replay(agent.call_state["call_id"])))
client.post("/hangup")

# 5. Spam, abuse and connect never trigger it.
for reason, key in [("sales call", "spam"), ("abusive caller after one warning", "abuse")]:
    client.post("/start", json={})
    stubbed([reply(tool_calls=[tool("end_call", reason=reason, kind=key)])])
    r = client.post("/turn", json={"text": "buy my product"}).get_json()
    check(f"{key}: no follow-up, flat closing line",
          r["reply"] == agent.CLOSING_LINES[key]["english"] and not any(f.startswith("follow-up") for f in said_and_fixed(agent.call_state["call_id"])[1]))
r, _, _ = ring_turn("accept")
check("connect: no follow-up", r and r["outcome"] == "connected" and not any(
      row["kind"] == "fixed_line" and row["detail"].startswith("follow-up") for row in db.replay(r["call_id"])))

# The follow-up follows the caller's latest substantive turn (live test: an English start, then a
# Hinglish message, got the English follow-up under the old voice lock).
client.post("/start", json={})
agent.call_state.update(speech_lang="english")
stubbed([reply(tool_calls=[MSG()])], lang="hinglish")
r = client.post("/turn", json={"text": "usko bol dena maine call kiya tha"}).get_json()
check("English start, Hinglish message -> Hinglish follow-up", r["reply"] == agent.pick_line(agent.FOLLOW_UP_LINES["message"], "hinglish"))
client.post("/hangup")
client.post("/start", json={})
agent.call_state.update(speech_lang="hinglish")
stubbed([reply(tool_calls=[MSG()])], lang="english")  # a short line misread as English keeps Hinglish (#242)
r = client.post("/turn", json={"text": "haan, bye"}).get_json()
check("short line after Hinglish -> follow-up stays Hinglish (#242)", r["reply"] == agent.pick_line(agent.FOLLOW_UP_LINES["message"], "hinglish"))
client.post("/hangup")
# Urdu or Devanagari script = Hinglish, whatever the judge says (eval #614, #615: judge said English).
for line, label in [("جتن کو بول دینا کہ شام کو جلدی گھر آ جائے", "Urdu script"), ("और बोलना कि दवाई लेते आना", "Devanagari")]:
    client.post("/start", json={})
    stubbed([reply(tool_calls=[MSG()])], lang="english")  # the judge gets it wrong
    r = client.post("/turn", json={"text": line}).get_json()
    check(f"{label} line, judge says English -> Hinglish follow-up", r["reply"] == agent.pick_line(agent.FOLLOW_UP_LINES["message"], "hinglish")
          and agent.call_state["lang"] == "hinglish")
    client.post("/hangup")

# Family call gets the follow-up too, then the family neutral line.
client.post("/start", json={})
agent.call_state.update(family=True)
stubbed([reply(tool_calls=[tool("take_message", message="Come home soon.")])])
r = client.post("/turn", json={"text": "tell him to come home"}).get_json()
check("family -> family follow-up line", r["reply"] == agent.FOLLOW_UP_LINES["family_message"]["english"])
stubbed([reply(tool_calls=[tool("end_call", reason="done", kind="done")])])
r = client.post("/turn", json={"text": "no that's it"}).get_json()
check("family after follow-up -> family neutral line", r["reply"] == agent.CLOSING_LINES["family_after_follow_up"]["english"])

# 6. Silence / hang-up after the follow-up.
client.post("/start", json={})
stubbed([reply(tool_calls=[REM()])])
client.post("/turn", json={"text": "vps renewal"})
line = agent.caller_silent()
summary = agent.finish_call()
check("silence after the follow-up -> neutral closing line", line == agent.CLOSING_LINES["after_follow_up"]["english"])
check("silence -> outcome logged as saved (reminder)", summary["outcome"] == "reminder")
check("silence on an already-ended call -> nothing said", agent.caller_silent() is None)
client.post("/start", json={})
stubbed([reply(tool_calls=[MSG()])])
client.post("/turn", json={"text": "tell him i called"})
r = client.post("/hangup").get_json()
check("hang-up after the follow-up -> outcome stays message", r["outcome"] == "message")
ev = [(row["kind"], row["detail"]) for row in db.replay(r["call_id"])]
check("hang-up logged as an event", ("caller_hung_up", "caller hung up") in ev)
check("hang-up: no closing line logged (nobody heard it)",
      [d for k, d in ev if k == "jarvis_said"][-1] == agent.FOLLOW_UP_LINES["message"]["english"])

# Bye with nothing saved: the standalone rule applies whether or not anything was saved.
check("SYSTEM_PROMPT has the standalone bye rule", "- They say bye or they're done: end_call." in agent.SYSTEM_PROMPT)
client.post("/start", json={})
made = stubbed([reply(tool_calls=[tool("end_call", reason="caller said bye", kind="done")])])
r = client.post("/turn", json={"text": "actually never mind, bye"}).get_json()
check("bye with nothing saved -> end_call -> 'nothing saved' closing line, one model request",
      r["ended"] and r["reply"] == agent.CLOSING_LINES["nothing_saved"]["english"] and len(made) == 1)
check("bye with nothing saved -> no follow-up, outcome abandoned",
      r["outcome"] == "abandoned" and not any(f.startswith("follow-up") for f in said_and_fixed(r["call_id"])[1]))

# Family: same standalone bye rule, outside the after-save line.
check("FAMILY_PROMPT has the standalone bye rule", "They say bye or they're done: end_call." in agent.FAMILY_PROMPT)
client.post("/start", json={})
agent.call_state.update(family=True)
made = stubbed([reply(tool_calls=[tool("end_call", reason="caller said bye", kind="done")])])
r = client.post("/turn", json={"text": "kuch nahi, bye"}).get_json()
check("family bye with nothing saved -> end_call -> 'family, nothing' closing line, one model request",
      r["ended"] and r["reply"] == agent.CLOSING_LINES["family_nothing"]["hinglish"] and len(made) == 1)  # Hinglish bye (#962 fix)
check("family bye with nothing saved -> no follow-up",
      not any(f.startswith("follow-up") for f in said_and_fixed(r["call_id"])[1]))

# The reworded after-save instruction: no self-reference, and a recital of it is caught.
_after_save = "Don't ask if there's anything else; it's asked for you."
check("after-save instruction is in both prompts", all(_after_save.lower() in p.lower() for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))
check("after-save instruction trips no self-reference", not agent.SELF_REFERENCE.search(_after_save))
check("reciting the after-save instruction is caught (hint_overlap)",
      all(agent.leak_check(_after_save, agent.state_hint(), p) == "hint_overlap" for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))

# Wording checks.
check("the follow-up question appears in neither prompt",
      all(q not in p for q in ("Anything else you want me to tell him", "Aur kuch batana hai unhe")
          for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))
check("no 'bye' instruction left for the model in either prompt",
      all("friendly bye" not in p and "Never say bye before" not in p for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))
check("every follow-up line passes leak_check against the live hint and both prompts",
      all(agent.leak_check(line, agent.state_hint(), p) is None
          for row in agent.FOLLOW_UP_LINES.values() for line in row.values() for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))
# Run B: the #244/#271 placeholder is written by code now; the model can't recite what it never sees.
check("placeholder text is in no prompt, schema or hint the model reads",
      all(agent.NO_REASON_TEXT not in text for text in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT, agent.state_hint(),
                                                     json.dumps(agent.tools + agent.family_tools))))
check("no 'the caller' / \"caller's\" in any tool schema (19 before run B)",
      not __import__("re").search(r"the caller|caller's", json.dumps(agent.tools + agent.family_tools), flags=2))

print("step 4: injected notes are second-person imperatives")
import re as _re, inspect as _inspect
# Every string the model reads mid-call: the hint (all notes switched on), every tool result and ERROR
# line in agent.py, and the two retry notes. None may describe the system in the third person.
agent.call_state.update(rings=1, family=False, reason_known=True)
_src = _inspect.getsource(agent)
_notes = [agent.state_hint(), agent.LEAK_RETRY_NOTE]
_notes += _re.findall(r'return \(?f?"((?:ERROR|Noted|Not connected|Jatin|Already|Reminder|Message|Call ended)[^"]*)"', _src)
_notes += _re.findall(r'"content": "([^"]*)"', _src)
THIRD_PERSON = _re.compile(r"\bthe caller\b|\bcaller's\b|\bthe user\b|\bthe assistant\b|\bthe model\b", _re.I)
check(f"found the injected notes to audit ({len(_notes)})", len(_notes) >= 15)
for note in _notes:
    check(f"no third person: {note[:60]}", not THIRD_PERSON.search(note) and not agent.SELF_REFERENCE.search(note))
# Final sweep (step 4 freeze): both full prompts too.
for _name in ("SYSTEM_PROMPT", "FAMILY_PROMPT"):
    _hits = THIRD_PERSON.findall(getattr(agent, _name)) + [m.group(0) for m in agent.SELF_REFERENCE.finditer(getattr(agent, _name))]
    check(f"no third person anywhere in {_name} ({len(_hits)} hits)", not _hits)
# Notes never say "callback"; prompts may still name what a CALLER asks for ("wants a callback"),
# but never offer "message or callback".
check("no injected note mentions a 'callback' (invites \"he'll call you back\")",
      not any("callback" in n.lower() for n in _notes))
check("neither prompt offers 'message or callback'",
      not any("message or callback" in p.lower() for p in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT)))

print("run B: no_reason_given")
client.post("/start", json={})
real_verdict, real_ask = agent.current_verdict, agent.ask_owner
briefings = []
agent.current_verdict = lambda: None                         # the judge says: reason NOT known
agent.ask_owner = lambda briefing: briefings.append(briefing) or False
agent.call_state.update(wants_jatin_turn=1, caller_agreed=True, reason_known=False, turns=3)
check("reason unknown, no flag -> lock refuses (as before)",
      agent.try_reach_owner("Rohit", "wants to talk") == agent.REFUSAL_RESULTS["reason"])
check("reason unknown, no_reason_given=true -> lock treats it as known, Jatin is rung",
      agent.try_reach_owner("Rohit", no_reason_given=True).startswith("Not connected") and len(briefings) == 1)
check("briefing carries the code-written placeholder", briefings and briefings[0].endswith(agent.NO_REASON_TEXT))
agent.current_verdict, agent.ask_owner = real_verdict, real_ask
r = agent.add_reminder("Rohit", "low", no_reason_given=True)
saved = db.reminders_for_call(agent.call_state["call_id"])
check("add_reminder with no_reason_given -> saved with the placeholder, model wrote no reason",
      r.startswith("Reminder saved") and saved and saved[-1][2] == agent.NO_REASON_TEXT)
check("no reason and no flag -> refused", agent.add_reminder("Rohit", "low").startswith("ERROR"))
client.post("/hangup")

print("saved notes are written to Jatin (run B leftover)")
import run_evals
both = [t for t in agent.tools + agent.family_tools if t["function"]["name"] == "take_message"]
check("normal take_message uses the note-to-Jatin description; family has its own content-only one (2026-11-26)",
      len(both) == 2 and both[0]["function"]["parameters"]["properties"]["message"] is agent.MESSAGE
      and agent.MESSAGE["description"].startswith("A note to Jatin, written to him")
      and both[1]["function"]["parameters"]["properties"]["message"]["description"].startswith("Only what they want Jatin to know"))
check("the schema's own example reads as a clean note",
      not run_evals.THIRD_PERSON_NOTE.search("Priya called; the invoice is late."))
for bad in ["They asked you to tell Jatin that they called.",                       # 506, 507
            "Called about the site and asked me to tell you that he called.",       # 483
            "They want you on the phone right now; they need to speak with you."]:  # 507
    check(f"third-person note flagged: {bad[:45]}", bool(run_evals.THIRD_PERSON_NOTE.search(bad)))
for good in ["Priya called; wants you to know she called.", "Papa called; come home early today.",
             "Rahul called about the VPS renewal; please call him back."]:
    check(f"note to Jatin passes: {good[:45]}", not run_evals.THIRD_PERSON_NOTE.search(good))

print("voice gender (prepared, off by default)")
check("default voice set is Sia (ElevenLabs, female, Sia 05 for both languages)",
      os.getenv("TTS_VOICE_SET") in (None, "", "sia") and voice.gender() == "female"
      and voice.voice_set()["provider"] == "elevenlabs" and set(voice.voice_set()["voices"].values()) == {voice.SIA05})
check("both voices in the Sia set are female, Sarvam fallback switched off",
      voice.VOICE_SETS["sia"]["gender"] == "female" and voice.SARVAM_FALLBACK_ON is False)
check("every voice set has exactly one gender", all(v["gender"] in ("male", "female") for v in voice.VOICE_SETS.values()))
real_gender = voice.gender
voice.gender = lambda: "male"
check("male voice: both prompts exactly as before",
      agent.prompt_for_voice(agent.SYSTEM_PROMPT) == agent.SYSTEM_PROMPT and agent.prompt_for_voice(agent.FAMILY_PROMPT) == agent.FAMILY_PROMPT)
TABLES = {"closing": agent.CLOSING_LINES, "follow-up": agent.FOLLOW_UP_LINES, "leak fallback": agent.LEAK_FALLBACK_LINES,
          "note saved": {"x": agent.LINES_NOTE_SAVED}, "note not saved": {"x": agent.LINES_NOTE_NOT_SAVED},
          "hold": {"x": server.HOLD_LINES}}
masc = run_evals.MASCULINE_FIRST_PERSON
check("male voice: Hinglish fixed lines unchanged", agent.pick_line(agent.CLOSING_LINES["message"], "hinglish") == "Theek hai, main bata dunga. Bye.")
voice.gender = lambda: "female"
try:
    for name, table in TABLES.items():
        for key, lines in table.items():
            if masc.search(lines["hinglish"]):
                check(f"{name}/{key}: gendered Hinglish line has a female variant", "hinglish_female" in lines)
            check(f"{name}/{key}: female voice speaks no masculine form", not masc.search(agent.pick_line(lines, "hinglish")))
            check(f"{name}/{key}: English line is the same for both genders", agent.pick_line(lines, "english") == lines["english"])
    for base in (agent.SYSTEM_PROMPT, agent.FAMILY_PROMPT):
        fp = agent.prompt_for_voice(base)
        check("female prompt: rule present once, masculine example gone",
              fp.count(agent.FEMININE_RULE) == 1 and "samajh gaya" not in fp and "samajh gayi" in fp or ("samajh" not in base and fp.count(agent.FEMININE_RULE) == 1))
        check("female prompt: reciting the rule is caught by the leak guard",
              agent.leak_check(agent.FEMININE_RULE[2:], agent.state_hint(), fp) == "hint_overlap")
        new_lines = [lines["hinglish_female"] for table in TABLES.values() for lines in table.values() if "hinglish_female" in lines]
        check(f"female prompt: all {len(new_lines)} new female fixed lines pass leak_check",
              all(agent.leak_check(line, agent.state_hint(), fp) is None for line in new_lines))
        check("female prompt: a normal feminine reply passes leak_check",
              agent.leak_check("Theek hai, main Jatin ko bata dungi.", agent.state_hint(), fp) is None)
    check("masculine check: flags dunga / karta hoon / raha hoon, passes dungi / karti hoon / third person",
          all(masc.search(t) for t in ["main bata dunga", "main try karta hoon", "connect kar raha hoon", "Samajh gaya."])
          and not any(masc.search(t) for t in ["main bata dungi", "main try karti hoon", "kya ho raha hai",
                                                "wo abhi baat nahi kar pa raha", "woh call karenge"]))
finally:
    voice.gender = real_gender

# Legacy speak(): the female Sarvam fallback stays in the code, switched off (Jatin, 2026-11-24).
real_el, real_sv = voice._speak_elevenlabs, voice._speak_sarvam
el_calls, sv_voices = [], []
voice._speak_elevenlabs = lambda text, vid: el_calls.append(vid) or (None, None, None, None)
voice._speak_sarvam = lambda text, lang, voices: sv_voices.append(voices[lang][0]) or (b"mp3", voices[lang][0], 1, 2)
try:
    real_close_tts()
    check("fallback OFF: ElevenLabs fails -> no audio, no Sarvam", real_speak("Theek hai.", "hinglish")[0] is None and sv_voices == [])
    voice.SARVAM_FALLBACK_ON = True
    real_close_tts()
    real_speak("Theek hai.", "hinglish"); real_speak("Okay.", "english")
    check("fallback ON (kept for later): female Sarvam voice, sticky for the call",
          sv_voices == ["simran_enhi_customer", "simran_enhi_customer"] and len(el_calls) == 2)
finally:
    voice.SARVAM_FALLBACK_ON = False
    voice._speak_elevenlabs, voice._speak_sarvam = real_el, real_sv
    real_close_tts()

print("fixed-line cache (step A)")
plan = voice.cache_plan(server.server_fixed_lines())
check("cache plan: every fixed line in its own language, once",
      {t for t, l in plan if l == "english"} == server.server_fixed_lines()["english"]
      and {t for t, l in plan if l == "hinglish"} == server.server_fixed_lines()["hinglish"] and len(plan) == len(set(plan)))
check("openers are cached as Hinglish only", all((o, "english") not in plan and (o, "hinglish") in plan for o in (agent.OPENER, *(f["hinglish"] for f in agent.FAMILY_OPENERS.values()))))
check("cache file names differ by language (tag) and by text",
      voice.cache_path("Okay.", "english") != voice.cache_path("Okay.", "hinglish") != voice.cache_path("Okay!", "hinglish"))
missing = [(t, voice.stream_voice(l)["name"]) for t, l in plan if not os.path.exists(voice.cache_path(t, l))]
check(f"every fixed line has a cached file (Sia 05, right language) ({len(plan) - len(missing)}/{len(plan)})", not missing)
for t, name in missing[:5]:
    print(f"       missing {name}: {t[:60]}")

print("pipeline fixes (live call #677)")
warmed.clear()
r = client.post("/start", json={}).get_json()
for _ in range(100):  # the pool runs two warm-ups (voice, speech-to-text): wait for the voice one itself
    if warmed:
        break
    time.sleep(0.01)
check("start opens the ElevenLabs connection ahead of the first reply", len(warmed) >= 1)  # earlier calls may add more
check("no stale-code warning when nothing changed", "warning" not in r)
client.post("/client_timing", json={"to_reply_text": 9100.4, "to_first_sound": 9600})
ct = [json.loads(row["detail"]) for row in db.replay(agent.call_state["call_id"]) if row["kind"] == "client_timing"]
check("page timing logged (release -> reply text, -> first sound)", ct and ct[-1]["to_reply_text"] == 9100 and ct[-1]["to_first_sound"] == 9600)
client.post("/hangup")
server.LOADED["agent.py"] -= 10   # pretend agent.py changed after the server started
r = client.post("/start", json={}).get_json()
check("code changed after start -> warning on the page + stale_code event",
      "agent.py" in r.get("warning", "") and any(row["kind"] == "stale_code" for row in db.replay(agent.call_state["call_id"])))
server.LOADED["agent.py"] += 10
client.post("/hangup")

print("invented-claims grader (eval)")
graded = []
real_grade = run_evals.grade_reply
run_evals.grade_reply = lambda reply, allowed: graded.append((reply, allowed)) or (["jatin: he's in tech"] if "tech" in reply else [])
client.post("/start", json={})
agent.client.chat.completions.create = scripted([judge(), reply("He's in tech, builds apps. Any message?")])
client.post("/turn", json={"text": "what does jatin do?"})
cid = agent.call_state["call_id"]
client.post("/hangup")
fails = run_evals.claim_failures(cid)
check("grader: only model-written replies are graded (opener and fixed lines are code's)",
      [r for r, _ in graded] == ["He's in tech, builds apps. Any message?"])
check("grader: the reply is graded against the caller's words", "Caller said: what does jatin do?" in graded[0][1])
check("grader: an unsupported claim fails the call", len(fails) == 1 and "he's in tech" in fails[0])
run_evals.grade_reply = real_grade

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
