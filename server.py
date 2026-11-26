# A small web server so a browser can talk to Jarvis.
# Week 2: typed turns (/turn), spoken turns (/turn_audio, step 3), every reply as text plus audio
# (step 4: Sarvam streaming TTS), and Jatin's phone (step 5: /ring and /answer).
# Step A (2026-11-24): two ElevenLabs voices (Sia 06 Hinglish + pre-lock, Sia 05 English). A reply's
# audio is no longer sent inside the JSON: the reply carries an audio_url, and /speech streams PCM from
# the fixed-line cache or live from ElevenLabs, so the browser starts playing at the first chunk.
#
# Run:   python server.py      then open http://localhost:5000
#
# One call at a time. agent.py keeps the call in one global dictionary (call_state), so only one
# request may touch it at once: /start, /turn, /turn_audio and /hangup all hold call_lock.
# The server is threaded only so that /ring and /answer can get through WHILE a turn is waiting
# for Jatin to pick up. Those two never touch call_state; they only use the `phone` slot below.
# Consequence (deliberate): /hangup also waits for the lock, so the caller can't hang up during a ring.
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
import json
import queue
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import access
import agent
import db
import voice

app = Flask(__name__, static_folder="static")


# ---------- running under a path (JARVIS_BASE_PATH, e.g. /jarvis; empty on a laptop) ----------
def base_path():
    base = (os.getenv("JARVIS_BASE_PATH") or "").strip().strip("/")
    return "/" + base if base else ""


class BasePath:
    """Serves the app under JARVIS_BASE_PATH whether or not the proxy in front strips it: a request for
    /jarvis/start and one for /start (prefix already stripped) both reach the /start route. The page gets a
    <base href> and uses relative URLs, so every fetch, audio URL and static file follows the base path."""

    def __init__(self, wsgi):
        self.wsgi = wsgi

    def __call__(self, environ, start_response):
        base = base_path()
        if base:
            path = environ.get("PATH_INFO", "")
            if path == base or path.startswith(base + "/"):
                environ["PATH_INFO"] = path[len(base):] or "/"
            environ["SCRIPT_NAME"] = base
        return self.wsgi(environ, start_response)


app.wsgi_app = BasePath(app.wsgi_app)

# Stale-code guard (2026-11-25): the live call #677 ran code from before the last two batches, because
# the server was started earlier and Python loads modules once. Any change to these files after start
# shows up as a warning on the page and a "stale_code" event in the call's log.
import os
CODE_FILES = ["agent.py", "voice.py", "server.py", "db.py", "audio_level.py"]
_HERE = os.path.dirname(os.path.abspath(__file__))
LOADED = {name: os.path.getmtime(os.path.join(_HERE, name)) for name in CODE_FILES}


def changed_since_start():
    return [name for name in CODE_FILES if os.path.getmtime(os.path.join(_HERE, name)) != LOADED[name]]
background = ThreadPoolExecutor(max_workers=2)  # connection warm-ups at the start of a call
active = {"call": False}  # is a call in progress right now?
call_lock = threading.Lock()  # one call-changing request at a time

# Jatin's phone. While a turn waits for him, ring_owner fills this in; the page polls /ring to see it
# and posts /answer when he clicks. `answered` is a threading.Event: a flag one thread waits on
# (ring_owner, inside the turn) until another thread sets it (/answer).
phone = {"ring_id": 0, "briefing": None, "hold_audio_url": None, "answer": None, "answered": None}

# Spoken by code (not the model) while the phone rings, so the caller doesn't hear dead air.
# Says only what is true: he's being tried. No promise he'll answer.
HOLD_LINES = {
    "english": "One sec, trying him now.",
    "hinglish": "Ek second, main try karta hoon.",
    "hinglish_female": "Ek second, main try karti hoon.",  # follows the voice's gender (agent.pick_line)
}


@app.get("/")
def page():
    """The page, with <base href> set to the base path so its relative URLs work under it."""
    html = open(os.path.join(_HERE, "static", "index.html"), encoding="utf-8").read()
    html = html.replace("<head>", f'<head>\n<base href="{base_path()}/">', 1)
    return Response(html, mimetype="text/html", headers={"Cache-Control": "no-store"})


# ---------- access code (JARVIS_ACCESS_CODE; see access.py) ----------
# Open without a session: the page itself, static files, the lock screen's status check and the unlock. EVERY
# other endpoint (anything that can reach OpenAI, ElevenLabs or Sarvam, and everything else) needs a session,
# checked here before the route runs, so a refused request never reaches a paid service.
OPEN_ENDPOINTS = {"page", "static", "session_status", "unlock"}


def set_cookie(response, name, value, max_age=None):
    response.set_cookie(name, value, max_age=max_age, httponly=True, secure=True, samesite="Lax",
                        path=request.script_root or "/")


def current_visitor():
    """Who is calling, for the per-visitor cap: the session's id, or "local" when there is no lock."""
    return access.session_visitor(request.cookies.get(access.SESSION_COOKIE)) or "local"


@app.before_request
def require_session():
    if not access.is_locked() or request.endpoint in OPEN_ENDPOINTS or request.endpoint is None:
        return None
    if access.session_visitor(request.cookies.get(access.SESSION_COOKIE)) is None:
        return jsonify(error="Locked. Enter the access code.", locked=True), 401
    return None


@app.get("/session")
def session_status():
    """The lock screen's first question: is this browser unlocked? Also hands out the client cookie that the
    wrong-code limit counts against. Never returns or logs the code."""
    unlocked = not access.is_locked() or access.session_visitor(request.cookies.get(access.SESSION_COOKIE)) is not None
    demo = access.demo_mode()
    response = jsonify(locked=not unlocked, demo=demo, notice=access.DEMO_NOTICE if demo else None)
    if access.is_locked() and access.client_id(request.cookies.get(access.CLIENT_COOKIE)) is None:
        set_cookie(response, access.CLIENT_COOKIE, access.new_client()[1], max_age=7 * 24 * 3600)
    return response


@app.post("/unlock")
def unlock():
    """Body: {"code": "..."} typed on the lock screen. Right code: a signed session cookie, valid 12 h.
    Wrong code: counted (5 per browser per 10 minutes, 30 for everyone together)."""
    if not access.is_locked():
        return jsonify(ok=True)
    cid = access.client_id(request.cookies.get(access.CLIENT_COOKIE))
    if cid is None:
        return jsonify(error="Please reload the page and try again."), 400
    wait = access.wrong_tries_wait(cid)
    if wait:
        return jsonify(error=f"Too many tries. Please wait about {wait // 60 + 1} minutes.", wait=wait), 429
    if not access.code_matches((request.get_json(silent=True) or {}).get("code")):
        access.note_wrong_try(cid)
        return jsonify(error="That code isn't right."), 401
    response = jsonify(ok=True)
    set_cookie(response, access.SESSION_COOKIE, access.new_session()[1], max_age=access.SESSION_SECONDS)
    return response


@app.post("/start")
def start():
    """Begin a new call. Body: {"caller_id": "1111"} or {} for an unknown caller."""
    visitor = current_visitor()
    cap = access.call_cap_hit(visitor)  # checked before anything that could cost money
    if cap:
        return jsonify(limit=cap, error=LIMIT_MESSAGE), 429
    with call_lock:
        if active["call"]:
            agent.finish_call(hung_up=True)  # a new call replaces an unfinished one
            voice.close_tts()
        access.count_call(visitor)
        body = request.get_json(silent=True) or {}
        caller_id = body.get("caller_id", "").strip()
        opener = agent.start_call(caller_id=caller_id, caller_lines=None, owner_answers=ring_owner, verbose=False)
        agent.begin_reply(None, body.get("reply_id"))
        if opener is None:  # family call: Jatin picked up straight away
            result = agent.finish_call()
            return jsonify(reply=None, ended=True, outcome=result["outcome"], call_id=result["call_id"])
        active["call"] = True
        # Open the ElevenLabs connection now, while the cached opener plays: the first streamed reply
        # then skips the ~1.2 s TLS handshake (#677 turn 1: first audio 1506 ms cold vs ~300 ms warm).
        background.submit(voice.warm_tts)
        background.submit(voice.warm_stt)
        stale = changed_since_start()
        warning = {}
        if stale:
            agent.record("stale_code", f"server started before these files changed: {', '.join(stale)}")
            print(f"WARNING: {', '.join(stale)} changed since this server started. Restart it.")
            warning = {"warning": f"Server is running old code ({', '.join(stale)} changed since it started). Restart server.py."}
        return jsonify(reply=opener, ended=False, call_id=agent.call_state["call_id"], **warning,
                       filler_after_ms=filler_after_ms(),
                       **spoken(opener, lang="hinglish"))


def filler_after_ms():
    """JARVIS_FILLER_AFTER_S in .env (default 2.0): how long after the caller lets go, with no audio yet, the page
    plays a filler. Read at each call start, so a new value needs only a restart. 2.0 = a quick "hmm" on most
    turns; 7.2 = about 1 turn in 4 (eval #1372-1425)."""
    try:
        return int(float(os.getenv("JARVIS_FILLER_AFTER_S", "2.0")) * 1000)
    except ValueError:
        return 2000


@app.get("/caller_ids")
def caller_ids():
    """The family caller IDs for the page's picker, labelled from FAMILY_NAMES in .env at request time (2026-11-26).
    The names live only in .env: the page fetches them, so they never appear in static/index.html or the repo."""
    # Demo mode: "Family (1111)", never the FAMILY_NAMES names (agent.caller_label).
    return jsonify(family=[{"id": n, "label": agent.caller_label(n)} for n in sorted(agent.FAMILY_IDS)])


@app.get("/filler")
def filler():
    """The page asks for this when no audio has started 1.5 s after the caller let go (once per turn, never on a
    fast turn: the page decides). A short cached line in the call's language, picked by situation (agent.next_filler:
    none after "anything else?", "working" after a ring request/yes/question, else an acknowledgement).
    Never synthesised live (a filler must be instant) and never put in the model's history. No lock: read + log."""
    headers = {"X-Sample-Rate": str(voice.PCM_RATE), "Cache-Control": "no-store"}
    if not active["call"]:
        return Response(b"", mimetype="audio/pcm", headers=headers)
    lang = agent.call_state.get("speech_lang") or agent.call_state.get("lang") or "hinglish"
    line, kind = agent.next_filler(lang)
    if line is None:  # skipped (the situation, random, or one played on the previous turn): silence, the reply follows
        db.log(agent.call_state.get("call_id"), "filler_skipped",
               json.dumps({"turn": agent.call_state.get("turns", 0), "kind": kind}))
        return Response(b"", mimetype="audio/pcm", headers=headers)
    pcm = voice.cached_pcm(line, lang)
    db.log(agent.call_state.get("call_id"), "filler",
           json.dumps({"turn": agent.call_state.get("turns", 0), "line": line, "kind": kind, "cached": pcm is not None}))
    return Response(_in_chunks(pcm or b""), mimetype="audio/pcm", headers=headers)


@app.post("/client_timing")
def client_timing():
    """The page's own clock (2026-11-25): from releasing the talk button to the reply text arriving, and to
    the first sound playing. The server's "waited" can't see upload or playback; this can. (No lock: log only.)"""
    body = request.get_json(silent=True) or {}
    fields = {k: int(body[k]) for k in ("to_reply_text", "to_first_sound") if isinstance(body.get(k), (int, float))}
    if fields and agent.call_state.get("call_id"):
        db.log(agent.call_state["call_id"], "client_timing", json.dumps({"turn": agent.call_state.get("turns", 0), **fields}))
    return jsonify(ok=True)


@app.post("/turn")
def turn():
    """One typed caller turn. Body: {"text": "hi, this is priya", "reply_id": "..."}. Returns Jarvis's reply.
    With ?stream=1 the reply arrives line by line as it's written (see locked_turn)."""
    started = time.time()
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    def work(emit, streaming):
        if not text:
            return None
        if turn_cap_hit():
            return end_for_turn_cap(emit, streaming, started, body.get("reply_id"))
        return run_turn(text, started, emit=emit, streaming=streaming, reply_id=body.get("reply_id"))
    return locked_turn(work, "Say something first.")


DIDNT_CATCH = "Sorry, I didn't catch that. Can you say it again?"


@app.post("/turn_audio")
def turn_audio():
    """One spoken caller turn: a recorded clip in the form field "audio" (and "reply_id").
    The clip is transcribed in memory and never saved. ?stream=1: line by line (see locked_turn)."""
    started = time.time()  # "caller waited" runs from here (they've stopped talking) to first audio
    clip = request.files.get("audio")
    audio = clip.read() if clip is not None else None
    filename, mimetype = (clip.filename or "turn.webm", clip.mimetype or "audio/webm") if clip is not None else (None, None)
    reply_id = request.form.get("reply_id")
    try:  # the clip's length, measured by the page (press -> release + tail): over 30 s goes to Scribe
        seconds = float(request.form.get("seconds"))
    except (TypeError, ValueError):
        seconds = None

    def work(emit, streaming):
        if audio is None:
            return None
        if turn_cap_hit():  # before transcription: no paid request after a cap
            return end_for_turn_cap(emit, streaming, started, reply_id)
        text, ms = voice.transcribe(audio, filename, mimetype, seconds=seconds)
        # One line per turn: which engine, how long, and why Scribe ran if it did.
        engine = voice.last_stt["engine"] or "none"
        fallback = f" ({voice.last_stt['why']})" if voice.last_stt["why"] else ""
        # The clip itself (#912: a cut-off clip couldn't be told from the log): its length and size.
        fallback += f" [clip {seconds:.1f} s, {len(audio) // 1024} KB]" if seconds is not None else f" [clip {len(audio) // 1024} KB]"
        if text is None or voice.not_speech(text):
            # Transcription failed, heard nothing, or heard only filler ("hmm") or the old hint echoed: ask again.
            # The model never sees it. The call carries on; nothing to save yet.
            why = "FAILED" if text is None else ("heard nothing" if not text else f"not speech: {text[:80]!r}")
            agent.record("stt", f"{engine} {ms}ms{fallback}, {why}")
            agent.begin_reply(say_sink(emit, started) if streaming else None, reply_id)
            agent.say_line(DIDNT_CATCH)  # history = what the caller heard
            return dict(heard=None, reply=DIDNT_CATCH, ended=False,
                        **({} if streaming else spoken(DIDNT_CATCH, started)))  # in the call's voice
        agent.record("stt", f"{engine} {ms}ms{fallback}")
        emit({"type": "heard", "text": text})
        # The model and the page get the transcript as is: Sarvam's is Roman already (no display rewrite).
        reply = run_turn(text, started, stt_ms=ms, emit=emit, streaming=streaming, reply_id=reply_id)
        return dict(heard=text, **reply)
    return locked_turn(work, "No audio received.")


# ---------- usage caps (access.py: calls per day, per visitor per day, turns per call) ----------
LIMIT_MESSAGE = "Demo limit reached for today. Please try again tomorrow."
# Said by code when a call reaches its turn cap; cached, so it costs nothing. No gendered word.
LIMIT_LINES = {"english": "Sorry, this demo call has reached its limit. Thank you for calling, bye.",
               "hinglish": "Sorry, is demo call ki limit poori ho gayi. Call karne ke liye shukriya, bye."}


def turn_cap_hit():
    return agent.call_state.get("turns", 0) >= access.max_turns_per_call()


def end_for_turn_cap(emit, streaming, started, reply_id):
    """The call reached JARVIS_MAX_TURNS_PER_CALL: a fixed line from the cache, the call ends, no model,
    transcription or live speech request."""
    agent.record("limit", f"turn cap reached ({access.max_turns_per_call()} turns)")
    agent.begin_reply(say_sink(emit, started) if streaming else None, reply_id)
    line = agent.say_line(agent.pick_line(LIMIT_LINES, agent.fixed_line_lang()))
    audio = {} if streaming else spoken(line, started)
    agent.call_state["ended"] = True
    active["call"] = False
    result = agent.finish_call()
    voice.close_tts()
    return dict(heard=None, reply=line, ended=True, limit="turns", message="Demo limit reached for this call.",
                outcome=result["outcome"], call_id=result["call_id"], **audio)


def locked_turn(work, missing_input):
    """Run one turn holding call_lock. work(emit, streaming) returns the reply as a dict, or None if the
    request had no input.
    Plain request: one JSON reply when the turn is done (typed turns, the tests).
    ?stream=1 (2026-11-25, the page): NDJSON, one event per line, sent as it happens:
      {"type": "heard", "text"} the transcript; {"type": "say", "text", "audio_url"} one line, cleared by the
      leak guards, its speech ready to fetch; {"type": "done", ...} the
      same fields as the plain reply. The turn runs on a worker thread that holds the lock until it ends,
      so /barge_in (no lock) can cut it short."""
    streaming = request.args.get("stream") == "1"
    call_lock.acquire()
    if not active["call"]:
        call_lock.release()
        return jsonify(error="No call in progress. Start a call first."), 400
    if not streaming:
        try:
            result = work(lambda event: None, False)
        finally:
            call_lock.release()
        return jsonify(result) if result is not None else (jsonify(error=missing_input), 400)

    events = queue.Queue()

    def worker():
        try:
            result = work(events.put, True)
            events.put({"type": "done", **result} if result is not None else {"type": "error", "error": missing_input})
        except Exception as error:  # run_turn already turns agent bugs into a fixed line; this is the rest
            events.put({"type": "error", "error": f"{type(error).__name__}: {str(error)[:120]}"})
        finally:
            call_lock.release()
            events.put(None)
    threading.Thread(target=worker, daemon=True).start()

    def body():
        while True:
            event = events.get()
            if event is None:
                return
            yield json.dumps(event) + "\n"
    return Response(body(), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store"})


def say_sink(emit, started):
    """Each line the agent clears to speak becomes a "say" event with its own speech URL, so the page
    starts fetching (and ElevenLabs starts rendering) it while the model writes the next one."""
    def sink(line):
        emit({"type": "say", "text": line, **spoken(line, started)})
    return sink


def run_turn(text, started, stt_ms=None, emit=None, streaming=False, reply_id=None):
    """Shared by typed and spoken turns: one caller turn through the agent.
    started: when the request arrived. stt_ms: transcription time (spoken turns only).
    streaming: each line goes out through emit as it's cleared; otherwise the whole reply gets one audio_url."""
    sink = say_sink(emit, started) if streaming else None
    try:
        reply = agent.handle_turn(text, sink=sink, reply_id=reply_id)
    except Exception as error:
        # A bug we didn't foresee: never leave the caller hanging. Same note-and-line path
        # as a provider outage, so the caller hears something true and Jatin gets a note.
        agent.record("server_error", f"{type(error).__name__}: {str(error)[:200]}")
        reply = agent.end_call_with_note(agent.call_state["messages"], "server_error")

    # registered now; /speech logs the voice stages when it plays. Nothing to play if a barge-in cut it all.
    audio = spoken(reply, started) if (reply and not streaming) else {}
    # Step 6: one timing line per turn. The voice stages (first audio, caller waited) are logged by
    # /speech as a "speech" event, because the audio now plays after this response is sent.
    agent.record_timing(stt=stt_ms)
    if agent.call_state["ended"]:
        active["call"] = False
        result = agent.finish_call()
        voice.close_tts()
        return dict(reply=reply, ended=True, outcome=result["outcome"], call_id=result["call_id"], **audio)
    return dict(reply=reply, ended=False, **audio)


@app.post("/barge_in")
def barge_in():
    """The caller pressed talk while Jarvis was still speaking or thinking (2026-11-25).
    Body: {"reply_id": "...", "heard": 2} = the reply being cut, and how many of its lines played (a line
    counts once half of it played). No lock: it has to get through while the turn it cuts holds the lock."""
    body = request.get_json(silent=True) or {}
    heard = body.get("heard")
    if isinstance(heard, bool) or not isinstance(heard, int) or heard < 0 or not body.get("reply_id"):
        return jsonify(error="reply_id and heard (a whole number, 0 or more) are required."), 400
    if not active["call"]:
        return jsonify(applied=False)
    return jsonify(**agent.barge_in(body["reply_id"], heard))


# ---------- speech: cached fixed lines and streamed replies ----------

speech_jobs = {}  # token -> one line waiting to be played: text, voice, call, when the caller stopped talking


def spoken(text, started=None, lang=None):
    """Register one line for the browser to play. lang decides how Sia says it (the accent tag is for
    English only): the caller's latest substantive language, unless given. Returns {"audio_url": ...};
    the audio itself is made when the page asks for it."""
    token = secrets.token_urlsafe(12)
    reply = agent.current_reply()
    speech_jobs[token] = {"text": text, "voice_lang": lang or agent.fixed_line_lang(),
                          "call_id": agent.call_state.get("call_id"), "turn": agent.call_state.get("turns", 0),
                          "started": started or time.time(), "cancel": reply["cancel"], "seq": max(len(reply["said"]) - 1, 0)}
    return {"audio_url": f"speech/{token}"}  # relative: resolved against the page's <base href>


def server_fixed_lines():
    """Every line code can speak: agent's fixed lines plus the hold line and "didn't catch that"."""
    lines = agent.all_fixed_lines()
    for lang in ("english", "hinglish"):
        lines[lang] |= {agent.pick_line(HOLD_LINES, lang), DIDNT_CATCH, agent.pick_line(LIMIT_LINES, lang)}
    return lines


def log_speech(job, **fields):
    db.log(job["call_id"], "speech", json.dumps({"turn": job["turn"], "seq": job.get("seq", 0), **fields}))


@app.get("/speech/<token>")
def speech(token):
    """Stream one line as 24 kHz 16-bit mono PCM. Cached fixed lines play at once; anything else streams
    live from ElevenLabs. If ElevenLabs fails before any audio: the cached outage line, and the call ends."""
    job = speech_jobs.pop(token, None)
    if job is None:
        return jsonify(error="No such line (already played?)."), 404
    headers = {"X-Sample-Rate": str(voice.PCM_RATE), "Cache-Control": "no-store"}
    voice_name = voice.stream_voice(job["voice_lang"])["name"]
    cancel = job.get("cancel") or threading.Event()
    if cancel.is_set():  # barge-in: the caller is talking; this line is never made
        log_speech(job, voice=voice_name, cancelled=True)
        return Response(b"", mimetype="audio/pcm", headers=headers)

    pcm = voice.cached_pcm(job["text"], job["voice_lang"])
    if pcm is not None:
        log_speech(job, voice=voice_name, cached=True, first=0, waited=int((time.time() - job["started"]) * 1000))
        return Response(_in_chunks(pcm, cancel), mimetype="audio/pcm", headers=headers)

    opened = voice.open_stream(job["text"], job["voice_lang"])
    if opened is None:
        return tts_outage(job, headers)
    pieces, voice_name, first_ms = opened
    waited = int((time.time() - job["started"]) * 1000)  # caller stopped talking -> first audio leaves the server

    def body():
        try:
            for piece in pieces:
                if cancel.is_set():  # barge-in mid-line: stop asking ElevenLabs for more
                    break
                yield piece
        finally:
            log_speech(job, voice=voice_name, cached=False, first=first_ms, waited=waited,
                       total=int((time.time() - job["started"]) * 1000))
    return Response(stream_with_context(body()), mimetype="audio/pcm", headers=headers)


def _in_chunks(pcm, cancel=None, size=9600):  # 0.2 s per chunk
    for start in range(0, len(pcm), size):
        if cancel is not None and cancel.is_set():
            return
        yield pcm[start:start + size]


def tts_outage(job, headers):
    """ElevenLabs failed twice before any audio (Jatin's rule, 2026-11-24): no backup voice. Same path as
    an OpenAI outage: save the note, play the cached outage line in the call's voice, end the call."""
    line, ended = agent.LINES_NOTE_NOT_SAVED["english"], {}
    if call_lock.acquire(timeout=15):  # a ring can hold the lock; then the call just isn't ended here
        try:
            if active["call"] and agent.call_state.get("call_id") == job["call_id"]:
                agent.record("tts_unavailable", "ElevenLabs failed twice before any audio")
                line = agent.end_call_with_note(agent.call_state["messages"], "tts_unavailable")
                result = agent.finish_call()
                active["call"] = False
                ended = {"X-Call-Ended": "1", "X-Outcome": result["outcome"], "X-Call-Id": str(result["call_id"])}
        finally:
            call_lock.release()
    log_speech(job, voice=voice.stream_voice(job["voice_lang"])["name"], failed=True)
    pcm = voice.cached_pcm(line, job["voice_lang"]) or b""
    return Response(_in_chunks(pcm), mimetype="audio/pcm", headers={**headers, **ended, "X-Reply": quote(line)})


# ---------- Jatin's phone ----------

def ring_owner(briefing):
    """agent.ask_owner calls this from inside a turn (or /start, for family). Rings the page's phone
    panel and waits for a click, at most RING_SECONDS. Returns "yes", "no", or "" for no answer."""
    hold = {"audio_url": None}
    if not agent.call_state["family"] or agent.call_state["rings"] >= 1:
        # A family call's FIRST ring happens before Jarvis has said anything, like a normal phone: no hold line
        # there. Its re-ring (asked for mid-call) gets one, like any ring during a conversation.
        line = agent.pick_line(HOLD_LINES, agent.fixed_line_lang())  # the call's voice language: it's cached
        agent.record("jarvis_said", f"(hold line) {line}")  # audit only: not in the model's history
        hold = spoken(line)

    answered = threading.Event()
    phone.update(ring_id=phone["ring_id"] + 1, briefing=briefing, hold_audio_url=hold["audio_url"],
                 answer=None, answered=answered)
    clicked = answered.wait(timeout=agent.RING_SECONDS)  # True if /answer set it, False on timeout
    answer = phone["answer"] if clicked else None
    phone.update(briefing=None, hold_audio_url=None, answered=None)  # stop ringing

    agent.record("ring", {"accept": "accepted", "decline": "declined"}.get(answer, f"no answer in {agent.RING_SECONDS}s"))
    return {"accept": "yes", "decline": "no"}.get(answer, "")


@app.get("/ring")
def ring():
    """Polled by the page while a request is pending: is Jatin's phone ringing? (No lock: read only.)"""
    if phone["answered"] is None:
        return jsonify(ringing=False)
    return jsonify(ringing=True, ring_id=phone["ring_id"], briefing=phone["briefing"],
                   hold_audio_url=phone["hold_audio_url"], seconds=agent.RING_SECONDS)


@app.post("/answer")
def answer():
    """Jatin clicks Accept or Decline. Body: {"answer": "accept"} or {"answer": "decline"}. (No lock.)"""
    choice = (request.get_json(silent=True) or {}).get("answer")
    if choice not in ("accept", "decline"):
        return jsonify(error="answer must be 'accept' or 'decline'."), 400
    answered = phone["answered"]
    if answered is None:
        return jsonify(error="The phone isn't ringing (it may have timed out)."), 409
    phone["answer"] = choice
    answered.set()  # wakes ring_owner up
    return jsonify(ok=True)


@app.post("/hangup")
def hangup():
    """The caller hangs up. Waits for the lock, so it can't land in the middle of a turn or a ring."""
    with call_lock:
        if not active["call"]:
            return jsonify(ended=True)
        active["call"] = False
        result = agent.finish_call(hung_up=True)
        voice.close_tts()
        return jsonify(ended=True, outcome=result["outcome"], call_id=result["call_id"])


if __name__ == "__main__":
    if voice.voice_set()["provider"] != "elevenlabs":
        raise SystemExit("server.py streams ElevenLabs PCM only; set TTS_VOICE_SET=sia (the default).")
    missing = [t for t, lang in voice.cache_plan(server_fixed_lines()) if voice.cached_pcm(t, lang) is None]
    if missing:
        print(f"WARNING: {len(missing)} fixed lines are not cached; they will stream live. Run: python prerender.py")
    # WhatsApp notes (phase 3): only the real server sends them, never tests or evals (they import this module but
    # don't run this block). Notes left in the outbox by an earlier run are retried now. The secret is never printed.
    import notify
    agent.on_note_saved = notify.enqueue
    if notify.start():
        left = len([r for r in notify.pending() if r["status"] == "pending"])
        print(f"WhatsApp notes: on ({left} waiting in the outbox)" + ("" if os.getenv("N8N_JARVIS_SECRET") else "  WARNING: N8N_JARVIS_SECRET is not set"))
    else:
        print("WhatsApp notes: off (N8N_JARVIS_WEBHOOK_URL not set)")
    if access.is_locked():
        print("Access code: on (the page is locked)" + ("" if access.secret_is_set() else
              "  WARNING: JARVIS_SESSION_SECRET is not set; sessions end when the server restarts"))
    else:
        print("Access code: off (no JARVIS_ACCESS_CODE)")
    print(f"Demo mode: {'on' if access.demo_mode() else 'off'} | caps: {access.max_calls_per_day()} calls/day, "
          f"{access.max_calls_per_visitor()} per visitor, {access.max_turns_per_call()} turns/call")
    host = os.getenv("JARVIS_HOST") or "127.0.0.1"   # localhost only by default: a proxy in front publishes it
    port = int(os.getenv("JARVIS_PORT") or "5000")
    print(f"Jarvis on http://{host}:{port}{base_path()}/  (provider: {agent.PROVIDER}, model: {agent.MODEL}, ring: {agent.RING_SECONDS}s)")
    app.run(host=host, port=port, threaded=True)
