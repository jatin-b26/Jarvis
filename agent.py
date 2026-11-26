import os
import re
import json
from dotenv import load_dotenv
import time
import threading
import random
from difflib import SequenceMatcher
from types import SimpleNamespace
import db
import voice  # only for the active voice's gender; agent.py never speaks
import notify
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ---------- provider: chosen in .env with LLM_PROVIDER=groq, openai or gemini ----------
# Both SDKs have the same shape: same client.chat.completions.create() call and the same
# error class names. So we import the classes under one set of names, and every retry and
# failure rule below works unchanged for either provider.
# max_retries=0 on every client: the SDKs retry twice by default, silently, underneath call_model's own
# retry loop. That hidden layer made outage behaviour impossible to reason about (one "attempt" could be
# 3 HTTP requests, each with its own wait). Now call_model's loop is the only retry logic there is.
PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
KEY_NAMES = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
MODEL_OPTIONS = {}  # extra settings sent with every model call (only Gemini uses any)
# Step B (2026-11-25): the MAIN model's reasoning effort, from .env (JARVIS_REASONING=low / minimal).
# Unset = the provider's default (gpt-5-mini: medium). The judge is not affected.

if PROVIDER in KEY_NAMES and not os.getenv(KEY_NAMES[PROVIDER]):
    # Fail at startup with a clear message, never mid-call.
    raise SystemExit(f"LLM_PROVIDER is '{PROVIDER}' but {KEY_NAMES[PROVIDER]} is missing from .env.")

if PROVIDER == "openai":
    from openai import (
        OpenAI, BadRequestError, RateLimitError, APIConnectionError, InternalServerError,
        APIStatusError,
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
    MODEL = "gpt-5-mini"         # the conversation
    # The small, fast classifier. A setting (2026-11-25): gpt-4.1-mini answers in ~0.9 s median against ~5 s for
    # gpt-5-nano, and matched the logged verdicts as well as the old judge matches itself (judge_agreement.py).
    # If it's retired, JARVIS_JUDGE_MODEL=gpt-4o-mini is the fallback.
    JUDGE_MODEL = os.getenv("JARVIS_JUDGE_MODEL", "gpt-5-nano")
elif PROVIDER == "groq":
    from groq import (
        Groq, BadRequestError, RateLimitError, APIConnectionError, InternalServerError,
        APIStatusError,
    )
    client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
    MODEL = "openai/gpt-oss-120b"        # the conversation
    JUDGE_MODEL = "openai/gpt-oss-20b"   # the small, fast classifier
elif PROVIDER == "gemini":
    # Google's OpenAI-compatible endpoint: same SDK, same error classes, just a different address.
    # FREE TIER (AI Studio, no billing): scripted eval scenarios only, never real calls, because its
    # terms may let Google train on prompts (README, Privacy).
    from openai import (
        OpenAI, BadRequestError, RateLimitError, APIConnectionError, InternalServerError,
        APIStatusError,
    )
    client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), max_retries=0,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    # Both models swappable from .env: the judge's speed matters as much as the conversation model's.
    MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")                    # or gemini-3.5-flash-lite
    JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")
    # Low thinking only: heavy reasoning (gpt-5-mini's 3-15 s) is exactly what we're trying to escape.
    MODEL_OPTIONS = {"reasoning_effort": "low"}
else:
    raise SystemExit(f"LLM_PROVIDER must be 'groq', 'openai' or 'gemini', not '{PROVIDER}'. Check your .env.")

# The conversation model can live on a different provider from the judge (main-model bake-off, 2026-11-25):
# JARVIS_MAIN_PROVIDER=groq|openai and JARVIS_MAIN_MODEL=<model>. Unset = same provider and model as above.
# Groq is reached through the OpenAI library at Groq's OpenAI-compatible address, so its errors are the same
# classes call_model already handles (a Groq rate limit is a RateLimitError like any other).
MAIN_PROVIDER = os.getenv("JARVIS_MAIN_PROVIDER", PROVIDER).strip().lower()
MAIN_BASE_URLS = {"groq": "https://api.groq.com/openai/v1", "openai": None}
if MAIN_PROVIDER == PROVIDER:
    main_client = client
elif MAIN_PROVIDER in MAIN_BASE_URLS:
    if not os.getenv(KEY_NAMES[MAIN_PROVIDER]):
        raise SystemExit(f"JARVIS_MAIN_PROVIDER is '{MAIN_PROVIDER}' but {KEY_NAMES[MAIN_PROVIDER]} is missing from .env.")
    from openai import OpenAI as _OpenAI
    main_client = _OpenAI(api_key=os.getenv(KEY_NAMES[MAIN_PROVIDER]), base_url=MAIN_BASE_URLS[MAIN_PROVIDER],
                          max_retries=0)
else:
    raise SystemExit(f"JARVIS_MAIN_PROVIDER must be 'openai' or 'groq', not '{MAIN_PROVIDER}'.")
MODEL = os.getenv("JARVIS_MAIN_MODEL") or MODEL

MAX_STEPS = 5

# How long Jatin's phone rings on the web page before it counts as no answer. From .env so a demo can
# use 10 s; in the terminal it isn't used (pressing Enter = no answer).
try:
    RING_SECONDS = int(os.getenv("RING_SECONDS", "30"))
except ValueError:
    print("RING_SECONDS in .env is not a whole number; using 30.")
    RING_SECONDS = 30
FAMILY_IDS = {"1111", "2222", "3333"}  # fake caller IDs of family members, for testing

OWNER_YES_WORDS = {
    "yes", "yeah", "yep", "yup", "ya", "ok", "okay", "sure", "fine", "done", "accept",
    "go", "ahead", "connect", "haan", "han", "hanji", "haanji", "ahann", "ahan", "hmm",
    "theek", "thik", "bilkul", "chalo", "y",
}
OWNER_NO_WORDS = {
    "no", "nope", "nah", "nahi", "nahin", "mat", "not", "dont", "don't", "busy",
    "later", "reject", "decline", "cut", "n",
}
RISKY_WORDS = ["otp", "password", "pin", "card", "money", "paisa", "paise", "upi", "transfer", "bank"]

call_state = {
    "ended": False,
    "turns": 0,
    "wants_jatin_turn": None,
    "caller_agreed": False,
    "reason_known": False,
    "rings": 0,
    "family": False,
    "caller_id": "",
    "lang": "english",
    "call_id": None,
    "caller_name": None,
    "outcome": None,
    "connected": False,
}


def record(kind, detail):
    """Write one line to the audit log, and show tool lines on screen."""
    db.log(call_state["call_id"], kind, detail)


# ---------- per-turn timing (step 6): where does a turn's time go? ----------

def new_timing():
    """Milliseconds spent in each stage this turn. model is a list: one entry per model call.
    judge = how long the judge took (in the background); judge_wait = how long the turn waited for it."""
    # first_say (2026-11-25, streaming): turn start -> first sentence cleared for speech, ring time excluded.
    return {"judge": 0, "judge_wait": 0, "model": [], "tools": 0, "ring": 0, "first_say": None}


def add_time(stage, started):
    """Add the time since `started` (a time.time() value) to this turn's `stage`. Returns the ms added."""
    ms = int((time.time() - started) * 1000)
    timing = call_state.setdefault("timing", new_timing())
    if stage == "model":
        timing["model"].append(ms)
    else:
        timing[stage] += ms
    return ms


def record_timing(**voice_stages):
    """Log this turn's timing as one event. server.py adds the voice stages
    (stt, tts_first, tts_total, waited); evals and the terminal log just the agent stages."""
    timing = dict(call_state.get("timing") or new_timing(), turn=call_state["turns"])
    timing.update({k: v for k, v in voice_stages.items() if v is not None})
    record("timing", json.dumps(timing))


# ---------- the judge: a small model that reads the call ----------

JUDGE_PROMPT = """You read one phone call and answer questions about it. Reply with JSON only, nothing else.

{
  "language": "english" or "hinglish",
  "caller_agreed_to_connect": true or false,
  "reason_known": true or false
}

language: the language Jarvis should reply in. "hinglish" if the caller's latest message is Hindi or Hinglish. Otherwise "english", including for any other language. If the latest message is too short to tell (like "ok" or "sure"), keep the language of their earlier messages, defaulting to english.

caller_agreed_to_connect: true if EITHER (a) Jarvis offered to try reaching Jatin and the caller has since said yes to being connected on this call, right now, in any wording or language ("sure", "ahan", "yeah?", "haan try karo", "please do"); OR (b) the caller themselves explicitly asks to be put through or for Jatin to be tried now, in any wording or language ("connect me", "put me through", "try him", "haan try kar lo", "unse baat karwa do"). A bare "haan", "ok" or "yes" with no offer before it and no request in it is NOT agreement. Asking for a callback, a message, or to be called later ("can he call me back?", "bol dena", "baad mein call kar lena") is NOT agreement: it is the opposite of being connected now. False if they declined, changed their mind, asked for a callback or message instead, or have not answered an offer.

reason_known: true only if the caller has said what the call is ABOUT ("it's about the invoice", "site ke baare mein"). Saying they want to talk to Jatin ("I want to talk to Jatin", "mujhe baat karni hai", "jatin hai kya") is NOT a reason. One exception: if the caller explicitly says there is no particular reason ("aise hi", "just wanna talk, nothing specific"), that counts as known. Merely stating they want to talk is not the same as saying there is no reason. False otherwise."""

def log_usage(model, response):
    """Record how many tokens one model call used, so eval runs show real cost."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return  # no usage info: skip logging rather than crash the call
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0  # part of "in" served from Groq's cache
    record("usage", f"model={model} in={usage.prompt_tokens} cached={cached} out={usage.completion_tokens}")


# HTTP errors that retrying won't fix, and the event name the replay shows for each.
NO_RETRY_STATUS = {
    401: "config_error",   # bad API key
    403: "config_error",   # key not allowed to use this model
    404: "config_error",   # model removed or misspelled
    413: "request_error",  # request too large (e.g. history grew too long)
    422: "request_error",  # request rejected as invalid
}


# Latency option 3 (2026-11-25, OFF unless JARVIS_CODE_LANGUAGE=1): code picks the reply/voice language from the
# caller's line, so the first audio no longer waits for the judge on every turn. The judge stays for the ring locks.
CODE_LANGUAGE = os.getenv("JARVIS_CODE_LANGUAGE") == "1"

MAIN_OPTIONS = ({"reasoning_effort": os.getenv("JARVIS_REASONING")}
                if os.getenv("JARVIS_REASONING") and MAIN_PROVIDER in ("openai", "groq") else {})


BILLING_SIGNS = ("insufficient_quota", "no credits remaining", "exceeded your current quota")


def is_billing_error(error):
    """A 429 that means the account is out of credits or quota, not a short rate limit."""
    code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if not code and isinstance(body, dict):
        code = body.get("code") or (body.get("error") or {}).get("code")
    text = str(error).lower()
    return code == "insufficient_quota" or any(sign in text for sign in BILLING_SIGNS)


def call_model(judge=False, **options):
    """Call the model, retrying on transient failures like rate limits.
    Returns None if the provider can't be used. Never raises for that.
    judge=True: a no-retry error (e.g. the judge model retired: 404 "deprecated") is logged as judge_model_error and
    does NOT mark the provider down: the call goes on without the judge (no ring) instead of ending."""
    if call_state.get("provider_down"):
        return None  # already failed on this call: don't make the caller wait through retries again
    delay = 2
    for attempt in range(5):
        try:
            api = client if judge or MAIN_PROVIDER == PROVIDER else main_client
            response = api.chat.completions.create(**({} if api is main_client and api is not client else MODEL_OPTIONS), **options)
            log_usage(options["model"], response)
            return response
        except (RateLimitError, APIConnectionError, InternalServerError) as error:
            if isinstance(error, RateLimitError) and is_billing_error(error):
                # Out of credits (2026-11-26: eval stopped at #1447, "You have no credits remaining"). Not a passing
                # rate limit: retrying only adds ~30 s of silence. Fail at once, like a bad key.
                call_state["provider_down"] = True
                record("billing_error", f"no retry: {options.get('model')}: {str(error)[:200]}")
                print(f"WARNING: billing problem on {options.get('model')}: credits or quota exhausted. Add credits.")
                return None
            if attempt == 4:
                call_state["provider_down"] = True
                if isinstance(error, RateLimitError):
                    call_state["provider_limited"] = True  # the provider's quota, not the model: evals re-run these
                record("provider_unavailable", f"{type(error).__name__} after 5 tries: {str(error)[:200]}")
                return None
            if call_state.get("verbose"):
                print(f"   [retry] {type(error).__name__}, waiting {delay}s")
            record("retry", f"{type(error).__name__}, waiting {delay}s")
            time.sleep(delay)
            delay = delay * 2
        except APIStatusError as error:
            # Errors that won't fix themselves in 30 seconds: no retries, end now.
            event = NO_RETRY_STATUS.get(error.status_code)
            if event is None:
                raise  # e.g. 400 BadRequestError: run_agent has its own handling for that
            if judge:
                hint = "deprecated or not found? set JARVIS_JUDGE_MODEL" if error.status_code in (403, 404) else "no retry"
                record("judge_model_error", f"judge model {options.get('model')} unavailable ({hint}): "
                                            f"{error.status_code} {str(error)[:160]}")
                print(f"WARNING: judge model {options.get('model')} unavailable ({error.status_code}): {hint}")
                return None
            call_state["provider_down"] = True
            record(event, f"{error.status_code} {type(error).__name__}, no retry: {str(error)[:200]}")
            return None

# Week 3: the judge runs ALONGSIDE the conversation model, not in front of it (WEEK3_PLAN.md).
# Its verdict only gates things inside the tools, which run after the model has decided, so the
# tools wait for it (current_verdict) and the locks still use THIS turn's verdict.
judge_pool = ThreadPoolExecutor(max_workers=1)


def judge_verdict(messages):
    """Runs in a background thread. Reads a copy of the call and returns (verdict, ms).
    verdict is None if the provider is down, {"failed": True} if the judge failed. Never raises,
    and never writes call_state's judge facts itself: the main thread applies them (apply_verdict)."""
    started = time.time()
    try:
        transcript = []
        for m in messages:
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role == "user" and content:
                transcript.append("Caller: " + content)
            elif role == "assistant" and content:
                transcript.append("Jarvis: " + content)
        response = call_model(
            judge=True,
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": "\n".join(transcript)},
            ],
            response_format={"type": "json_object"},
        )
        if response is None:  # provider down, or the judge model itself unavailable (judge_model_error)
            verdict = None if call_state.get("provider_down") else {"failed": True}
        else:
            verdict = json.loads(response.choices[0].message.content)
    except Exception:  # bad request, bad JSON, anything: a failed judge must not end the call
        verdict = {"failed": True}
    return verdict, int((time.time() - started) * 1000)


def start_judge(messages):
    """Start this turn's judge in the background, on a copy of the call so far."""
    call_state["judge_future"] = judge_pool.submit(judge_verdict, list(messages))


def current_verdict():
    """Wait for this turn's judge (usually it has already finished) and apply its verdict, once.
    Called by the tools that need it, and at the end of every turn."""
    future = call_state.get("judge_future")
    if future is None:
        return  # already applied this turn (or no judge started)
    call_state["judge_future"] = None
    started = time.time()
    verdict, judge_ms = future.result()  # judge_verdict never raises
    add_time("judge_wait", started)      # how long the turn actually stood still waiting for it
    call_state.setdefault("timing", new_timing())["judge"] = judge_ms
    apply_verdict(verdict)


def apply_verdict(verdict):
    """Main thread only. Turn the judge's verdict into call facts."""
    if verdict is None:
        return  # provider is down; run_agent ends the call cleanly
    if verdict.get("failed"):
        print("   [warn] judge failed, assuming no agreement yet")
        record("judge", "FAILED, assuming no agreement")
        return

    # A line in Arabic/Urdu or Devanagari script is Hindi/Urdu speech as transcribed: Hinglish, whatever
    # the judge says. (The judge called an Urdu-script line "English", so the follow-up came out in
    # English: eval #614, #615. Its prompt sends "any other language" to English.)
    if CODE_LANGUAGE:
        pass  # option 3: the language was set by code at the start of the turn (set_code_language)
    elif verdict.get("language") == "hinglish" or in_indic_script(latest_caller_line()) or has_hindi_words(latest_caller_line()):
        call_state["lang"] = "hinglish"
    else:
        call_state["lang"] = "english"

    # Some facts ACCUMULATE, some are LIVE.
    # reason_known accumulates: once the caller has said what the call is about, that doesn't un-happen,
    # so the judge can switch it on but never off (it used to flip back on later turns: calls #177, #182).
    # Cost, honestly: a judge false positive now sticks for the rest of the call too. Acceptable because
    # reason_known only unlocks a ring, and the call still connects only if Jatin says yes.
    if verdict.get("reason_known"):
        call_state["reason_known"] = True

    # caller_agreed is live: re-read every turn, because a caller can genuinely change their mind
    # ("caller reverses their decision" in the evals tests exactly that).
    if call_state["wants_jatin_turn"] is None:
        call_state["caller_agreed"] = False
    else:
        call_state["caller_agreed"] = bool(verdict.get("caller_agreed_to_connect"))

    record("judge", f"lang={call_state['lang']} agreed={call_state['caller_agreed']} reason_known={call_state['reason_known']}")

    # One voice for both languages now (Sia 05, 2026-11-24), so nothing is locked. Fixed lines follow the
    # language of the caller's latest SUBSTANTIVE turn: a caller who switches to Hinglish mid-call gets the
    # Hinglish follow-up (live test, 2026-11-24). A bare "bye"/"ok" keeps the previous language (#242: the
    # judge read a Hinglish caller's final "bye" as English).
    if not CODE_LANGUAGE and language_is_clear(latest_caller_line()) and call_state.get("speech_lang") != call_state["lang"]:
        call_state["speech_lang"] = call_state["lang"]
        record("speech_lang", call_state["lang"])


# Roman Hindi words that don't occur in English. 2+ different ones in a line = Hinglish, whatever the judge says.
# #962: the judge read "nahi bas itna hi, bye" as English; at 5 words it counted as a clear turn, so the goodbye
# came out in English. Checked on all 114 logged caller lines: no English line has 2 of these.
HINDI_WORDS = {
    "nahi", "nahin", "bas", "itna", "itni", "hai", "hain", "haan", "theek", "thik", "accha", "achha", "acha", "kya",
    "kyun", "kyon", "mein", "mujhe", "aap", "aapko", "unko", "usko", "unhe", "bol", "bolna", "bolo", "dena", "karo",
    "karna", "karni", "tha", "thi", "raha", "rahi", "hoon", "hu", "yaar", "ji", "abhi", "phir", "matlab", "chalo",
    "lekin", "aur", "baat", "kuch", "toh", "woh", "gaya", "gayi", "hua", "hogi", "hoga", "rehne", "dijiye", "sakte",
}


def has_hindi_words(text):
    return len(set(re.findall(r"[a-z']+", (text or "").lower())) & HINDI_WORDS) >= 2


def code_language(text):
    """The caller's language from their line alone (option 3), or None = keep the previous language.
    Devanagari/Urdu script or 2+ Hindi-only words: hinglish. Exactly 1 Hindi-only word ("mai Rohit hu",
    "haan try kar lo"): ambiguous, keep. No Hindi word and a clear turn (3+ words, not a greeting): english.
    Shorter ("bye", "ok"): keep. Checked on all 115 logged caller lines with a judge verdict: it disagrees with
    the judge's majority on 3, and on all 3 the code is right."""
    if in_indic_script(text):
        return "hinglish"
    hindi = len(set(re.findall(r"[a-z']+", (text or "").lower())) & HINDI_WORDS)
    if hindi >= 2:
        return "hinglish"
    if hindi == 0 and language_is_clear(text):
        return "english"
    return None


def set_code_language(text):
    """Option 3, at the start of a turn: the reply and voice language, before any model or judge call."""
    # A first line that isn't clear ("hello", "mai Rohit hu") defaults to Hinglish, not English: the opener is
    # Hinglish and most callers speak it; an English caller's clear 3+ word line switches at once (Jatin, 2026-11-25).
    lang = code_language(text) or call_state.get("speech_lang") or "hinglish"
    call_state["lang"] = lang
    if call_state.get("speech_lang") != lang:
        call_state["speech_lang"] = lang
        record("speech_lang", f"{lang} (code)")


def in_indic_script(text):
    """True if the line has Arabic/Urdu (U+0600-06FF, U+0750-077F, U+FB50-FEFF) or Devanagari
    (U+0900-097F) letters: how speech-to-text writes Hindi/Urdu speech."""
    return any("؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" or "ﭐ" <= ch <= "﻿"
               or "ऀ" <= ch <= "ॿ" for ch in text or "")


def latest_caller_line():
    for message in reversed(call_state.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":  # model turns are SDK objects
            return message.get("content") or ""
    return ""


def language_is_clear(text):
    """Enough words to tell English from Hinglish: not a bare greeting, and at least 3 words.
    ("hello", "haan ji", "ok sure" are not clear; "hi this is priya" is.)"""
    return not is_greeting_only(text) and len(text.split()) >= 3


# ---------- locks (code guarantees) ----------

def is_junk(value):
    return value.strip().lower() in [
        "", "?", "??", "unknown", "n/a", "none", "not provided", "unspecified",
    ]


# Written by code, never shown to the model: a placeholder the model sees is a placeholder it recites
# (#244, #271). The model sets no_reason_given; code writes this into the reminder and the briefing.
NO_REASON_TEXT = "wants to talk to Jatin, gave no other reason"


def reason_or_placeholder(reason, no_reason_given):
    return NO_REASON_TEXT if no_reason_given else (reason or "")


def check_ready(caller_name, reason):
    if is_junk(caller_name) or is_junk(reason):
        return "ERROR: name or reason missing. Ask for what's missing."
    if caller_name.strip().lower() == "unnamed caller" and call_state["turns"] < 3:
        return "ERROR: ask their name first. 'Unnamed caller' only if they refuse."
    return None


def remember_name(caller_name):
    if caller_name and not is_junk(caller_name):
        call_state["caller_name"] = caller_name


def note_what_we_heard(args):
    """Keep the latest name and reason the model passed to ANY tool, even a rejected one.
    Used only for the failed-call note, so a crash mid-call still leaves Jatin something."""
    if not isinstance(args, dict):
        return
    for key, state_key in [("caller_name", "heard_name"), ("reason", "heard_reason"), ("message", "heard_reason")]:
        value = args.get(key)
        if isinstance(value, str) and not is_junk(value):
            call_state[state_key] = value


def owner_says_yes(text):
    words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").split()
    if any(word in OWNER_NO_WORDS for word in words):
        return False
    return any(word in OWNER_YES_WORDS for word in words)


def ring_kind(answer):
    """"accepted", "declined" (an explicit no: the Decline button) or "no_pickup" (no answer: the ring timed out,
    or Enter in the terminal). Family calls treat the last two differently (Jatin, 2026-11-26)."""
    if owner_says_yes(answer or ""):
        return "accepted"
    return "declined" if (answer or "").strip() else "no_pickup"


def family_names():
    """FAMILY_NAMES in .env, e.g. "1111:Papa,2222:Mummy" (private: never in the repo). Read at call time."""
    names = {}
    for part in os.getenv("FAMILY_NAMES", "").split(","):
        number, _, name = part.partition(":")
        if number.strip() and name.strip():
            names[number.strip()] = name.strip()
    return names


def demo_mode():
    """JARVIS_DEMO_MODE=1 (a public demo): family names never reach the page or the caller."""
    return os.getenv("JARVIS_DEMO_MODE") == "1"


def caller_label(number):
    """How the page shows a caller ID: "Papa (1111)" if FAMILY_NAMES maps it, else the number as it is.
    Demo mode: "Family (1111)" for every family number, never the mapped name."""
    number = (number or "").strip()
    if demo_mode() and number in FAMILY_IDS:
        return f"Family ({number})"
    name = family_names().get(number)
    return f"{name} ({number})" if name else number


def family_shown():
    """The family caller's name where the page or the model can see it (Jatin's phone panel, tool results):
    "Family" in demo mode, else the mapped name. Saved notes and WhatsApp keep the real name (family_name)."""
    return "Family" if demo_mode() else family_name()


def family_name():
    """The family caller's name for notes and Jatin's phone: the private mapping, else "Family member"."""
    return family_names().get(call_state.get("caller_id") or "", "Family member")


def ask_owner(briefing):
    record("owner_briefed", briefing)
    phone = call_state["owner_queue"]
    started = time.time()  # the ring wait is its own stage in the turn's timing
    if callable(phone):
        # The web page's "Jatin's phone" panel (server.py): rings, waits for a click,
        # returns "yes" (Accept), "no" (Decline) or "" (no answer in RING_SECONDS).
        answer = phone(briefing)
    else:
        print(f"\n   [Jatin's phone] {briefing}")
        answer = next_owner_answer("   [Jatin's phone] Connect? Say yes or no (Enter = didn't pick up): ")
    add_time("ring", started)
    kind = ring_kind(answer)
    call_state["last_ring"] = kind
    if kind == "declined":
        call_state["ever_declined"] = True  # family: no re-ring after a decline, ever
    record("owner_decision", f"answered '{answer}' -> {'ACCEPTED' if kind == 'accepted' else kind.replace('_', ' ')}")
    return kind == "accepted"


def state_hint():
    notes = []
    # The judge now runs alongside this model call, so this note can't use its language verdict
    # (it would be a turn late: a caller opening in Hindi would be told "reply in English").
    # The model reads the caller's message itself.
    # Step 4 (2026-11-23): every note is a second-person imperative. No "the caller", no description of
    # the system: a descriptive sentence reads fine as dialogue, and the model spoke paraphrases of these
    # notes to callers (#216, #219, #238). Remove the recitable text instead of forbidding the recitation.
    notes.append(
        # Step 3 wording, only "the caller's" -> "their" (2026-11-23). The step 4 rewrite dropped the "If"
        # and read as "use Hinglish": English callers got Hinglish in 4 of 90 calls, vs 0 of 45 before it.
        "Reply in their language. If their latest message is mostly Hindi or Hinglish (in any script), "
        "reply in casual spoken Hinglish, Roman letters only, simple everyday words, the way friends talk, "
        "never formal Hindi, never Devanagari. Otherwise reply in casual, friendly English; one word from "
        "another language doesn't count."
    )
    # Both openers say he can't pick up, so this is true on every call (it used to be a flag).
    notes.append("Don't say again that Jatin can't pick up.")
    if call_state["rings"] >= 1 and not call_state["family"]:
        notes.append("Jatin was already tried once. Don't offer to try again.")
    if call_state["reason_known"]:
        # Safe to use a turn late: reason_known only accumulates, so if it was true last turn it's true now.
        notes.append("Don't ask what it's about again.")
    # (The "you don't know what it's about yet, don't offer" note is gone: a turn late, it would say
    # "you don't know" on exactly the turn the caller gives the reason. That steering now lives in
    # caller_wants_jatin's result, which waits for this turn's verdict.)
    return " ".join(notes)


# ---------- tools (the real functions) ----------

def after_save(saved, kind):
    """Decide what follows a successful save. The FIRST save of a call always gets the code-spoken
    follow-up (run_agent says it once the tool batch is done; no model request). A save after the
    follow-up means they added something: the model confirms it briefly and the call waits for their
    bye, like any other turn. (It used to say "Now end_call.": the call then ended before they said
    bye, #381. Jatin's ruling (b), 2026-11-23.)
    (A caller_done field used to skip the follow-up when the model judged them already done. It
    guessed wrong on "just leave a message" / "bas bol dena": 14 of 21 save calls ended before the
    caller said bye, 2026-11-23. Fields that record what was said are fine; one that asks the model to
    predict the caller is finished is not, so it's gone.)"""
    if call_state["follow_up_spoken"]:
        # Never read it back (live test, 2026-11-24: the caller's own words were quoted back, with the wrong
        # person). A short confirmation is all the caller needs.
        # 2026-11-25: "...and wait." came back as "Waiting." / "I'll wait." (#724, #726): a short instruction
        # at the end of a result is a line the model can say. Every clause now names a thing NOT to do.
        # 2026-11-25 (Jatin): code confirms it (ADDED_LINES) and drops the model's text. The model's own
        # confirmations were where "saved exactly as you said" and "Waiting." came from (#724-726).
        call_state["added_due"] = True
        return f"{saved} They've heard it confirmed."
    call_state["follow_up_due"] = kind
    return saved


# Phase 3 (2026-11-26): server.py sets this to notify.enqueue, so a live call's saved notes go to Jatin's WhatsApp.
# Evals, tests and the terminal leave it None: they never POST to the real webhook.
on_note_saved = None


def note_saved(kind, caller_name, text, urgency=None):
    """A reminder or message was just saved: hand it to the notifier (outbox + background POST). Never raises,
    never waits on the network."""
    if on_note_saved is None:
        return
    counts = call_state.setdefault("notes_saved", {})
    counts[kind] = counts.get(kind, 0) + 1
    name = family_name() if call_state["family"] else caller_name
    try:
        on_note_saved(notify.build_note(call_state["call_id"], kind, counts[kind], name, call_state["family"], text, urgency))
    except Exception as error:
        record("notify_failed", f"not queued: {type(error).__name__}")


# Family reminders (Jatin, 2026-11-26): the model often fills the reason with the callback itself ("Call back",
# "Please call you back", "Requested callback"), which read "Papa wants a callback: Call back" on WhatsApp. Those words
# are dropped; what's left is the real reason ("when free", "about the bank"), or nothing.
CALLBACK_WORDS = re.compile(r"\b(call\s*back|callback|calls?|calling|ring|phone|get\s+back|reach\s+out)\b", re.I)
# Words that carry nothing once the callback words are gone ("Please call you back", "Request to call the caller").
EMPTY_WORDS = re.compile(
    r"\b(please|kindly|requests?|requested|wants?|wanted|asked|asks|ask|for|to|a|an|the|caller|them|they|me|you|him|"
    r"her|jatin|back|kar|karo|kare|karna|le|lo|lena|lijiye|do|dena|mujhe|unhe|usko|ko|se)\b", re.I)
LEADING_CALLBACK = re.compile(
    r"^\s*(please\s+)?((request(ed)?|wants?)\s+(to\s+)?)?(a\s+)?(call\s*back|callback|call|ring|phone)"
    r"(\s+(them|me|you|him|her|the\s+caller|back))*\s*[,;:-]?\s*", re.I)


def real_reason(reason):
    """The reason as given, minus a leading callback phrase; "" if it only restates the callback.
    "about the bank" -> unchanged; "Call them when free" -> "when free"; "Please call you back" -> ""."""
    if not reason or is_junk(reason):
        return ""
    reason = " ".join(reason.split()).strip(" .")
    if not CALLBACK_WORDS.search(reason):
        return reason
    if not re.search(r"[a-z]", EMPTY_WORDS.sub(" ", CALLBACK_WORDS.sub(" ", reason)), re.I):
        return ""
    return LEADING_CALLBACK.sub("", reason).strip(" .,;:-") or reason


def add_reminder(caller_name="", urgency="low", reason="", no_reason_given=False):
    if call_state["family"]:
        # Family (Jatin, 2026-11-26): no name or reason checks. The name comes from the caller ID mapping; the
        # reason is what they said, or just that they want a callback.
        who = family_name()
        reason = real_reason(reason) or "wants a callback"
        urgency = urgency if urgency in ("low", "medium", "high") else "low"
        db.save_reminder(call_state["call_id"], "reminder", who, reason, urgency)
        note_saved("reminder", who, reason, urgency)
        call_state["outcome"] = "reminder"
        return after_save(f"Reminder saved from {family_shown()}.", "reminder")
    reason = reason_or_placeholder(reason, no_reason_given)
    problem = check_ready(caller_name, reason)
    if problem:
        return problem
    remember_name(caller_name)
    db.save_reminder(call_state["call_id"], "reminder", caller_name, reason, urgency)
    note_saved("reminder", caller_name, reason, urgency)
    call_state["outcome"] = "reminder"
    return after_save(f"Reminder saved for {caller_name}.", "reminder")


# Duplicate saves (2026-11-25). A message is a repeat if it is nearly the same word sequence as one already
# saved on this call (SequenceMatcher ratio >= 0.8 on lowercased words) AND every word it adds is filler.
# #726: "Ramesh called; he says the site is ready to go live." vs the saved "Ramesh called; the site is
# ready to go live." -> ratio 0.9, adds only "he says": a repeat. "...ready to go live tomorrow" adds
# "tomorrow": new information, saved. A heuristic: tested on #709, #715, #724-728.
FILLER_WORDS = {
    "he", "she", "they", "says", "said", "say", "called", "calling", "wants", "want", "asked", "asks", "you",
    "your", "to", "that", "note", "please", "just", "again", "also", "and", "the", "a", "an", "is", "was",
    "has", "have", "jatin", "message", "exactly", "as", "it", "know", "tell",
}
DUPLICATE_RESULT = ("ERROR: duplicate of a note already saved on this call; nothing new was saved. "
                    "They've heard it's already with him.")


def _words(text):
    return re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))


# #1015/#1016: "save it again exactly how I said it" saved the caller's raw line ("hi, this is Ramesh, please
# tell Jatin the site is ready to go live") next to "Ramesh called; the site is ready to go live." The greeting
# and the self-introduction made it look new. So both notes are reduced to their CONTENT first: greetings, the
# caller's self-introduction, the caller's name and the filler words go, then the comparison runs as before.
GREETINGS = {"hi", "hello", "hey", "namaste", "namaskar"}
SELF_INTRO = re.compile(r"\b(this is|i am|i'm|it's|mera naam)\s+\w+(\s+hai)?\b|\bmain\s+\w+(\s+\w+)?\s+bol\s+(raha|rahi)\s+(hoon|hu|hun)\b"
                        r"|\b\w+\s+here\b")


def note_content(text, names=()):
    text = SELF_INTRO.sub(" ", (text or "").lower().replace("’", "'"))
    drop = GREETINGS | FILLER_WORDS | {n.lower() for n in names if n}
    return [w for w in _words(text) if w not in drop]


def repeats_a_saved_message(message, caller_name=None):
    names = set()
    for name in (caller_name, call_state.get("caller_name")):
        if name and not is_junk(name):
            names |= set(_words(name))
    new = note_content(message, names)
    for old_text in call_state.get("saved_messages", []):
        old = note_content(old_text, names)
        added = set(new) - set(old)
        if SequenceMatcher(None, old, new).ratio() >= 0.8 and added <= FILLER_WORDS:
            return True
    return False


def take_message(message, caller_name=None):
    if call_state["family"]:
        if is_junk(message):
            return "ERROR: no message yet. Ask what they want Jatin to know."
        # The private caller-ID mapping first, then a name they said, then "Family member" (Jatin, 2026-11-26).
        mapped = family_names().get(call_state.get("caller_id") or "")
        who = mapped or (caller_name if caller_name and not is_junk(caller_name) else "Family member")
    else:
        problem = check_ready(caller_name or "", message)
        if problem:
            return problem
        who = caller_name
    if repeats_a_saved_message(message, caller_name):
        # "Save it again exactly as I said" re-saved the same English note, then "saved exactly as you said"
        # (#724-726). Nothing new = nothing saved. The result names no action and has no line to say.
        record("duplicate_save", message)
        call_state["duplicate_due"] = True  # code says DUPLICATE_LINES; the model's text is dropped
        return DUPLICATE_RESULT
    remember_name(who)
    db.save_reminder(call_state["call_id"], "message", who, message)
    note_saved("message", who, message)
    call_state.setdefault("saved_messages", []).append(message)
    call_state["outcome"] = "message"
    kind = "family_message" if call_state["family"] else "message"
    return after_save(f"Message saved from {family_shown() if call_state['family'] else who}.", kind)


def caller_wants_jatin(**ignored):
    """The caller asked for Jatin or to be put through. Records it: try_reach_owner needs this first.
    (Saying he's unavailable is the opener's job now, fixed in code.)"""
    call_state["wants_jatin_turn"] = call_state["turns"]
    current_verdict()  # this turn's judge reading, before deciding what to tell the model
    if not call_state["reason_known"]:
        # Reason unknown: the offer is not shown at all. Removing the option beats instructing against it.
        # "yet" promised an offer later, and the model made it even after "can he call me back?" (#470, #563,
        # #564): so say what to do when they ask for something else first.
        return ("Noted. Ask what the call is about (and their name if you don't have it). Don't offer anything yet. "
                "If they then ask him to get back to them, or to pass on a message, save that.")
    # "try him" came out as "Jatin ko try karun" in 30 of 36 Hinglish calls shown the offer (a literal
    # translation). Say it as a call attempt, with the Hinglish shape in under 6 words (overlap guard).
    # The Hinglish shape goes in only for a Hinglish caller: an English caller who got it heard "call karne ka
    # try karoon?" (#932, #1000). No Hindi words in any tool result an English caller's turn reads.
    hinglish = '; in Hinglish: "call karne ka try"' if call_state.get("lang") == "hinglish" else ""
    return (f"Noted. Offer to try calling him (no promise he'll answer){hinglish}. "
            "Or offer a message or a reminder for him. Then wait for their answer.")


def refused(key):
    """A refused ring: code says REFUSAL_LINES[key] after the batch (the model's text that turn is dropped).
    #912 and #1062: after a refusal the model told the caller it was ringing. A phrase list is always one
    wording behind the model, so the model no longer writes that turn at all."""
    call_state["refusal_due"] = call_state.get("refusal_due") or key
    return REFUSAL_RESULTS[key]


def family_reach_owner():
    """A family caller asks for another try (Jatin, 2026-11-26). Locks in code: at most one re-ring per call, only
    after a no-pickup, never after a decline. Refusals are fixed lines; connecting still needs his yes."""
    if call_state.get("ever_declined"):
        return refused("family_declined")
    if call_state["rings"] >= 2:
        return refused("family_tried")
    # rings goes up AFTER the ring: server.ring_owner speaks the hold line for a family call once rings >= 1.
    accepted = ask_owner(f"Family call again: {family_shown()} (caller ID {call_state['caller_id']})")
    call_state["rings"] += 1
    if accepted:
        call_state["ended"] = True
        call_state["connected"] = True
        call_state["outcome"] = "connected"
        return "Jatin ACCEPTED. They've heard they're being connected."  # code says CONNECTED_LINES
    call_state["decline_due"] = f"family_{call_state['last_ring']}"  # code says FAMILY_RING_LINES[...]
    return "Not connected. They have heard that and been offered a reminder or a message. Save nothing until they answer."


def try_reach_owner(caller_name="", reason="", no_reason_given=False):
    if call_state["family"]:
        return family_reach_owner()
    current_verdict()  # the locks below must use THIS turn's verdict, never a stale one
    reason = reason_or_placeholder(reason, no_reason_given)
    if check_ready(caller_name, reason):
        name_missing = is_junk(caller_name or "") or (caller_name.strip().lower() == "unnamed caller" and call_state["turns"] < 3)
        if not name_missing and call_state["reason_known"]:
            # #1404: they had already answered ("aise hi" = no particular reason), but the call left out the reason.
            # A code line would re-ask a question they answered; the model fixes the arguments and calls again.
            return REASON_ARGUMENT_MISSING
        return refused("name" if name_missing else "reason")
    if call_state["wants_jatin_turn"] is None:
        return "ERROR: use caller_wants_jatin first, then wait for their answer."  # only a made-up call: no code line
    if not call_state["caller_agreed"]:
        return refused("not_agreed")
    # "No particular reason" is an answer: the prompt accepts it, so the lock does too. Without this the
    # judge (reason unknown) and the prompt (no reason is fine) disagreed and the lock sided with the
    # judge: 9 blocked attempts in "caller refuses to give a name", run #343-387.
    if not call_state["reason_known"] and not no_reason_given:
        return refused("reason")
    if call_state["rings"] >= 1:
        return refused("tried")
    remember_name(caller_name)
    call_state["rings"] += 1
    briefing = f"Incoming call from {caller_name}: {reason}"
    if any(word in reason.lower() for word in RISKY_WORDS):
        briefing += "\n   [Jatin's phone] WARNING: this mentions codes or money. Be careful."
    if ask_owner(briefing):
        call_state["ended"] = True
        call_state["connected"] = True
        call_state["outcome"] = "connected"
        return "Jatin ACCEPTED. They've heard they're being connected."  # code says CONNECTED_LINES
    # Code says the decline line (DECLINE_LINES) right after this batch: no model request. Live call 2026-11-25:
    # the model skipped "he couldn't take it" and went straight to "message ya reminder?".
    call_state["decline_due"] = True
    return "Not connected. They have heard that and been offered a message or a reminder. Save nothing until they answer."


END_KINDS = ("done", "spam", "abuse")


def end_call(reason, kind):
    if kind not in END_KINDS:
        return "ERROR: kind must be done, spam or abuse. Retry."
    call_state["ended"] = True
    call_state["end_reason"] = reason
    # The model's kind picks the closing line; no keyword match on its reason any more (#322: an OTP
    # scam ended with "scam" in the reason, which no keyword caught, and got the warm goodbye).
    call_state["end_kind"] = kind
    if kind == "spam":
        call_state["outcome"] = "spam"
    # The model no longer writes the goodbye: run_agent speaks a fixed closing line from code
    # (CLOSING_LINES). So this result needs no "don't say noted" warning any more.
    return f"Call ended: {reason}"


TOOL_FUNCTIONS = {
    "add_reminder": add_reminder,
    "take_message": take_message,
    "caller_wants_jatin": caller_wants_jatin,
    "try_reach_owner": try_reach_owner,
    "end_call": end_call,
}


# ---------- the menus the model sees ----------

def make_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


NAME = {"type": "string", "description": "Their real name, e.g. 'Priya'"}
REASON = {
    "type": "string",
    "description": (
        "Why they called, in one short summary in English. Translate if they spoke another language. "
        "Never copy their words in their own language. Leave out if no_reason_given is true."
    ),
}
# A saved message is a note Jatin reads, so it's written TO him. Run B saved "They asked you to tell Jatin
# that they called." (#506, #507) and "asked me to tell you that he called" (#483): third person, and
# "you" meant Jarvis. The example shows the shape (who called; what for him), in under 6 words a line.
MESSAGE = {
    "type": "string",
    "description": (
        "A note to Jatin, written to him as \"you\": who called, then what they want him to know, in one or "
        "two sentences in English. Example: \"Priya called; the invoice is late.\" Translate if they spoke "
        "another language. Never copy their words in their own language."
    ),
}
NO_REASON = {
    "type": "boolean",
    "description": "true only if you asked what it's about and they said there's no particular reason.",
}

end_call_tool = make_tool(
    "end_call",
    "Politely end the call. Use when they say goodbye or have nothing more to ask, "
    "for sales, spam or scam calls, or after one warning if they're abusive.",
    {
        "reason": {"type": "string", "description": "Why the call is ending, e.g. 'sales call'"},
        "kind": {"type": "string", "enum": list(END_KINDS), "description": (
            "done: they said bye or they're done. spam: sales, spam, or a scam (asking for codes or money). "
            "abuse: still abusive after one warning."
        )},
    },
    ["reason", "kind"],
)

tools = [
    make_tool(
        "add_reminder",
        "Save a reminder for Jatin to call them back. Use when they want a callback and it can wait.",
        {
            "caller_name": NAME,
            "reason": REASON,
            "no_reason_given": NO_REASON,
            "urgency": {"type": "string", "enum": ["low", "medium", "high"], "description": (
                "How urgent this is, judged by you from what they said. Never ask them to choose. "
                "If unclear, use low. high only for a genuine emergency or something that must be handled within the hour."
            )},
        },
        ["caller_name", "urgency"],
    ),
    make_tool(
        "take_message",
        "Save a message for Jatin when they only want to pass on information and need no callback.",
        {
            "caller_name": NAME,
            "message": MESSAGE,
        },
        ["caller_name", "message"],
    ),
    make_tool(
        "caller_wants_jatin",
        "Use when they ask for Jatin or to be put through to him, in any words "
        "(\"can I talk to Jatin\", \"put me through\"). Takes no arguments. Records the request, "
        "which try_reach_owner needs first, and tells you what to do next.",
        {},
        [],
    ),
    make_tool(
        "try_reach_owner",
        "Ring Jatin, tell him who is calling and why, and connect the call only if he says yes. "
        "Use ONLY once they have agreed to your offer to try him, or asked to be put through (after caller_wants_jatin), "
        "and you know their name and what the call is about (or they said there's no particular reason).",
        {"caller_name": NAME, "reason": REASON, "no_reason_given": NO_REASON},
        ["caller_name"],
    ),
    end_call_tool,
]

family_tools = [
    make_tool(
        "take_message",
        "Save the message a family member wants Jatin to get. Use as soon as they tell you anything to pass on. "
        "Their name is optional.",
        {
            # Family notes (Jatin, 2026-11-26): the WhatsApp summary already says who called ("🏠 Family call - Papa
            # left a message: ..."), so the note is only the content (a live note began "family called; they asked you...").
            "message": {"type": "string", "description": (
                "Only what they want Jatin to know, in their own words in Roman script, e.g. 'Ghar jaldi aa jao.' or "
                "'Come home early.' Never Devanagari. "
                "No lead-in like 'Family called' or 'They asked you': who called is shown separately.")},
            "caller_name": {"type": "string", "description": "Their name or relation if they said it, e.g. 'Papa'. Leave out if not said."},
        },
        ["message"],
    ),
    make_tool(
        "add_reminder",
        "Save a reminder only when they ask him to call them back (\"bolo mujhe call kar le\", \"remind kar do\"). "
        "A message to pass on (\"usko bolo ghar aa jaye\") is take_message, not a reminder.",
        {
            "reason": {"type": "string", "description": (
                "What they said it's about, one short summary in English. Leave out if they only want a callback.")},
            "urgency": {"type": "string", "enum": ["low", "medium", "high"], "description": (
                "Judged by you from what they said. Never ask. If unclear, low. high only for a real emergency.")},
        },
        [],
    ),
    make_tool(
        "try_reach_owner",
        "Ring Jatin again. Use only when they ask for another try. Takes no arguments. Whether it rings is "
        "decided for you, and what they hear after it is said for you.",
        {},
        [],
    ),
    end_call_tool,
]


# ---------- the rules the model reads (prompt = a request) ----------

SYSTEM_PROMPT = """
You are Jarvis, Jatin's AI assistant, answering his phone when he can't. You are not Jatin: never call them Jatin; if they think you are, say you're his assistant and he's not on this line. Your opener was already said; don't reintroduce yourself.

# Style
- A warm, polite phone assistant, not a call centre: casual, 1-2 short sentences, "you" and "I". Never repeat yourself.
- You have no life of your own: no opinions, no experiences, nothing you have seen, watched, followed or done. Never talk as if you had. Your one job is their message for Jatin.
- Ask as a request, not a demand: "Can I take your name?" not "Who's calling?".
- Hinglish asks stay polite and consistent: "aap" goes with "-iye" or "sakte hain", never with "ho".
- Keep every spoken line short: one idea, no extra clauses. "Theek hai." beats "Theek hai Rohit, main samajh gaya, aapki baat sun li."
- Use the language the per-turn note gives (default English). Hinglish: everyday words mixed with common English, like friends; never formal Hindi or Devanagari.

# Name and reason
- Get their name and what it's about. Ask only for what's missing. Never invent or guess values.
- "mai Bagga hu", "mera naam Bagga hai", "I am Bagga" are names.
- A greeting like "hello" is not a reason: ask for their name, use no tool.
- Wanting to talk to Jatin is not a reason: ask what it's about. No particular reason? Accept it, don't re-ask; set no_reason_given.
- "Unnamed caller" only after asking for the name twice unanswered.
- Never ask for their phone number: caller ID already has it.
- Frustrated caller: acknowledge briefly, then ask for the one missing thing.

# Actions
- Once you have their name and what it's about, offer the next step in that same reply, even if they never asked for Jatin: try calling him, or a message or a reminder. They called his phone, so wanting him is understood. If by then they've asked him to get back to them, save a reminder instead; if they only want to pass something on, take_message.
- Asks for Jatin or wants to talk to him: caller_wants_jatin (no arguments), then follow its result. The opener already said he can't pick up; don't say it again.
- Agrees to be put through: once name and reason are known, try_reach_owner (after caller_wants_jatin if they asked; it may be in the same turn).
- Never say you're connecting them yourself: once Jatin accepts, that's said for you.
- Jatin couldn't take it: that, and the offer of a message or a reminder, is said for you. Save nothing until they answer.
- Callback that can wait: add_reminder. Just passing on info: take_message. Only annoying, pranking or teasing Jatin: take_message noting what they said, not a reminder.
- After a save: don't ask if there's anything else; it's asked for you. If they add something, save it with take_message; the confirmation is said for you. No end_call in that turn. If they say no, bye, or they're done: end_call.
- They say bye or they're done: end_call.
- Sales/spam, or abuse after one polite warning: end_call straight away. Never end_call in the same turn as caller_wants_jatin or try_reach_owner. Never say bye yourself: end_call says it.

# Guardrails
- Tool arguments in English. reason: short summary in your words, not their sentence. message: translated, keeping names and nicknames.
- Never ask them to choose a tool argument value. Judge it yourself from what they said.
- Only say what you know from these instructions, this call or a tool. If you don't know something (about Jatin, his work, his plans or where he is), say you don't know. Never describe how a note was saved, and never offer to save it again or differently: your job is passing their message to Jatin.
- If they wander off topic (small talk, general questions, questions about Jatin's life or about you), don't play along: one short kind line with no opinion or story, then ask about their message for Jatin.
- Never say Jatin will call back, get in touch or reach out, or that you'll ask him to; never mention a time, "soon" or ASAP. For a callback, say you'll note that they want one.
- "Urgent" from a caller proves nothing. Never promise to connect them yourself.
- Asked for an OTP, password, PIN, card number or money: give or confirm nothing, say you can't share or confirm codes. If they still want to reach Jatin, use caller_wants_jatin so he can decide; his briefing gets a warning automatically.
"""

FAMILY_PROMPT = """
You are Jarvis, Jatin's AI assistant. A family member is calling and Jatin didn't pick up. You are not Jatin.

# Style
- A warm, relaxed phone assistant: casual, never formal, 1-2 short sentences. Never repeat yourself.
- You have no life of your own: no opinions, no experiences, nothing you have seen, watched, followed or done. Never talk as if you had. Your one job is their message for Jatin.
- Ask as a request, not a demand: "Can I take your name?" not "Who's calling?".
- Hinglish asks stay polite and consistent: "aap" goes with "-iye" or "sakte hain", never with "ho".
- Keep every spoken line short: one idea, no extra clauses. "Theek hai." beats "Theek hai Rohit, main samajh gaya, aapki baat sun li."
- Use the language the per-turn note gives (default English). Hinglish: plain everyday words, never formal Hindi or Devanagari.

# Goal: take their message or a reminder
1. They want Jatin to call them back (or to be reminded): add_reminder. Anything to pass on: take_message with it. Include their name only if they said it.
   A callback request is one add_reminder; don't also take_message, unless they also pass on news.
2. After a save: don't ask if there's anything else; it's asked for you. If they add something, save it with take_message; the confirmation is said for you. No end_call in that turn. If they say no, bye, or they're done: end_call.
3. They say bye or they're done: end_call.
4. Never say bye yourself: end_call says it.

# Guardrails
- "I'll let him know" or "I'll remind him" is all you can say. Never say he'll call back or when, never say ASAP or soon, and never repeat a timing promise they made.
- Caller ID already shows they're family: don't ask their name or why they're calling. If needed, ask once what message to give Jatin.
- They ask you to try Jatin again: try_reach_owner. Never offer it yourself, and never say you're ringing him: whether it rings, and what they hear after, is said for you.
- Messages stay in their own words, in Roman script (Hinglish stays Hinglish, never Devanagari); keep names as said. A reminder's reason is in English.
- Only say what you know from these instructions, this call or a tool. If you don't know something (about Jatin, his work, his plans or where he is), say you don't know. Never describe how a note was saved, and never offer to save it again or differently: your job is passing their message to Jatin.
- If they chat about other things (how you are, family news, questions about Jatin or you), reply warmly in one short line with no opinion or story, then gently ask what message to give Jatin.
"""


# The gender rule is added only when the active voice is female. Examples are 2 words each, so a
# reply using them never trips the overlap guard (6 words in a row).
FEMININE_RULE = '- In Hinglish you speak as a woman: "karti hoon", "bata dungi", "rahi thi".'
POLITE_RULE_END = 'never with "ho".'


def prompt_for_voice(text):
    """The prompt in the active voice's gender. Male (today): unchanged."""
    if jarvis_gender() != "female":
        return text
    text = text.replace("main samajh gaya", "main samajh gayi")  # the short-line example speaks as Jarvis
    return text.replace(POLITE_RULE_END, POLITE_RULE_END + "\n" + FEMININE_RULE, 1)


# Openers: fixed in code, so every caller hears them exactly. "AI assistant" is a disclosure, not style:
# keep it. Spoken by Sia 06 (the pre-lock voice) in Hinglish, her strength; most callers speak Hinglish.
# "Jatin ki assistant": the possessive follows the assistant's (female) gender. Approved by Jatin 2026-11-24.
# Watch: English-caller drift (a Hinglish opener is the first line in the model's history). Fallback if it
# drifts: the mostly-English opener ("May I know your name, and what you'd like to talk to him about?").
OPENER = ("Hello, Jarvis this side, Jatin ki AI assistant. Woh abhi phone nahi utha sakte. "
          "Aap apna naam aur kis baare mein baat karni thi, bata sakte hain?")
# Family openers (Jatin, 2026-11-26): the ring happens before Jarvis speaks, so the opener says how it went. Keeps the
# AI disclosure. Never "declined" to family. Hinglish only, like OPENER (Jatin dropped the English versions).
FAMILY_OPENERS = {
    "declined": {
        "hinglish": "Hello, Jarvis this side, Jatin ki AI assistant. Jatin abhi call nahi le paaye. "
                    "Unhe callback ke liye remind kar doon, ya koi message dena hai?",
    },
    "no_pickup": {
        "hinglish": "Hello, Jarvis this side, Jatin ki AI assistant. Jatin ne abhi phone nahi uthaya. "
                    "Ek baar aur try karoon, ya remind kar doon, ya koi message?",
    },
}


# ---------- the agent loop ----------

GREETING_WORDS = {
    "hello", "hi", "hey", "hii", "hiii", "helo", "hola", "namaste", "namaskar",
    "ji", "bhai", "sir", "madam", "good", "morning", "afternoon", "evening", "there",
}


def is_greeting_only(text):
    words = text.lower().replace(",", " ").replace("!", " ").replace(".", " ").replace("?", " ").split()
    return len(words) > 0 and all(word in GREETING_WORDS for word in words)

def looks_like_a_tool_call(text):
    for name in TOOL_FUNCTIONS:
        if name + "(" in text:
            return True
    return False


# ---------- leak guard (2026-11-23) ----------
# Catches the model speaking its own instructions or notes to the caller ("The assistant must reply now
# in Hinglish."). Two pattern nets: HEURISTICS, not guarantees. Tuned offline on calls #204-252 and
# every spoken line in the db (982): 10 hits, all real leaks, zero false positives. A catch is logged
# (event "leak_guard", with which check fired), the line is never said, and the model gets one retry.
SELF_REFERENCE = re.compile(
    # Third person about a part of the system. The determiner matters: bare "assistant" is in the opener.
    r"\bthe\s+(assistant|caller|user|model|system)\b"
    # "assistant" + a modal verb, e.g. "(assistant couldn't reply in two voices...)" (#242).
    r"|\bassistant\s+(must|should|could|couldn['’]?t|can['’]?t|cannot|will|won['’]?t|would|needs?)\b",
    re.IGNORECASE,
)


def six_word_runs(text):
    words = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
    return {tuple(words[i:i + 6]) for i in range(len(words) - 5)}


def leak_check(text, hint, prompt):
    """Which check fired ("self_reference" or "hint_overlap"), or None if the line looks clean.
    hint_overlap = 6 words in a row shared with this turn's hint or the active system prompt
    (#244 read the prompt's placeholder reason aloud). 5 words fires on "can I take your name"."""
    if SELF_REFERENCE.search(text):
        return "self_reference"
    if six_word_runs(text) & (six_word_runs(hint) | six_word_runs(prompt)):
        return "hint_overlap"
    return None


# ---------- claim guard (2026-11-25, live call #912) ----------
# Jarvis said "main Jatin ko abhi ring karti hoon" and "dobara ring karti hoon" while nothing rang: the lock had
# refused the ring, and the eval's connecting check only ever SCORED finished calls, never guarded a live one
# (and it didn't know Hinglish). Now every sentence is checked before it's spoken:
#   claim:       a statement that Jarvis is ringing, calling, trying, connecting or putting them through, now or
#                next, in English or Hinglish. Never the model's to say: a real ring and a real connection are
#                announced by code (the hold line, CONNECTED_LINES). Questions are offers, and pass.
#   early_offer: a question offering to ring/try/connect before caller_wants_jatin has fired AND this turn's
#                judge says the reason is known (#679, #794, #884, #912: offered before the reason).
# Patterns: HEURISTICS on the phrasings seen so far, not a guarantee. Run on every past model line (2026-11-25)
# before going live; the eval counts and lists every block so a wrong one is visible.
ACTION_CLAIMS = [re.compile(p) for p in (
    # English, first person, now or next
    r"\b(i'll|i will|i'm going to|i am going to|let me|i'm|i am)\s+(now\s+|just\s+|quickly\s+|going\s+to\s+)?"
    r"(ring|call|dial|phone|connect|transfer|put you through|try (him|jatin|calling|ringing|reaching|to (call|reach|ring|connect)))",
    r"\b(ringing|calling|dialing|dialling|connecting|trying)\s+(him|jatin|you)\b",
    r"\bputting you through\b",
    # Hinglish, first person: ring karti hoon / call kar rahi hoon / connect kar deti hoon / try karungi ...
    r"\b(ring|call|phone|try|connect|transfer|dial)\s+(kar\s+|karke\s+|karne\s+)?(raha|rahi|rha|rhi)\s+(hoon|hu|hun|hoo)\b",
    r"\b(ring|call|phone|try|connect|transfer|dial)\s+(karta|karti)\s+(hoon|hu|hun|hoo)\b",
    r"\b(ring|call|phone|try|connect|transfer|dial)\s+(kar|karke)\s+(deta|deti|dunga|dungi|doonga|doongi)\b",
    r"\b(ring|call|phone|try|connect)\s+(karunga|karungi|karoonga|karoongi)\b",
    r"\b(call|phone)\s+(lagata|lagati|lagaunga|lagaungi|laga\s+(raha|rahi|deta|deti))\b",
    r"\bbaat\s+karwa\s+(raha|rahi|deta|deti|dunga|dungi)\b",
    # "koshish" = trying (#1062: "main unko call karne ki koshish karti hoon", the #912 class in new words)
    r"\bkoshish\s+(kar\s+)?(raha|rahi|rha|rhi)\s+(hoon|hu|hun)\b",
    r"\bkoshish\s+(karta|karti)\s+(hoon|hu|hun)\b",
    r"\bkoshish\s+(karunga|karungi|karoonga|karoongi)\b",
    r"\bkoshish\s+kar\s+(deta|deti|dunga|dungi)\b",
)]
# Hold phrases: blocked whenever the model says them, question or not. A real ring happens inside the tool call,
# so while the model is speaking nothing is ringing; the real hold line is spoken by code.
HOLD_PHRASES = re.compile(r"\bthoda\s+intezaar\s+(kijiye|kariye|karein|karo)\b|\bplease\s+hold\b|\bhold\s+on\b"
                          r"|\bhold\s+the\s+line\b|\bstay\s+on\s+the\s+line\b|\bline\s+pe\s+rahiye\b")
# Masculine first person (the voice is female): "karta hoon", "samajh gaya", "-unga". Not "raha ... tha": that is
# also third person ("Rohit bula raha tha", #771). The eval uses this same pattern.
MASCULINE_FIRST_PERSON = re.compile(
    r"\b(karta|sakta|raha|deta|leta|rakhta|chahta|samajhta|jaanta|bolta|dekhta)\s+(hoon|hu|hun|hoo)\b"
    r"|\b\w+(unga|oonga)\b|\bsamajh gaya\b|\bmain\b[^.?!]{0,40}\bpa raha\b", re.I)
RING_OFFERS = [re.compile(p) for p in (
    r"\b(try|ring|call|phone|reach|dial)\w*\s+(him|jatin)\b",
    r"\btry\s+(calling|ringing|reaching|to\s+(call|reach|ring|connect))\b",
    r"\bput(ting)?\s+you\s+through\b", r"\bconnect(ing)?\s+you\b", r"\btransfer\w*\s+you\b",
    r"\b(jatin|unhe|unko|usko|unse)\b[^?]{0,30}\b(call|ring|try|connect|phone)\b",
    r"\bcall\s+karne\s+ka\s+try\b", r"\bconnect\s+kar\s*(doon|du|dun|dein|doo)\b",
    r"\btry\s+(karoon|karun|karu|kar\s+loon|kar\s+lu|kar\s+doon)\b",
)]


# Checked on all 3,673 past model lines (2026-11-25): without these, conditional offers said as statements
# ("I can try putting you through once I have your name", #225, #763) and a true report after a real ring
# ("I tried calling him", #595) read as claims.
OFFER_WORDS = re.compile(r"\b(can|could|if|once|before|need|want me to|would you|shall|should)\b")  # English only
NOT_A_CLAIM = re.compile(r"\b(can't|cannot|couldn't|won't|unable|not able|nahi|nahin|tried|try kiya|kiya tha|ki thi)\b")


def is_question(sentence):
    return sentence.rstrip(" \"'”’)").endswith("?")


def _said(sentence):
    return sentence.lower().replace("’", "'")


def is_offer(said):
    """A question, or an English statement shaped as an offer or a condition ("I can try...", "if you want...")."""
    return is_question(said) or bool(OFFER_WORDS.search(said))


def claims_action(sentence):
    """True if the sentence says Jarvis is ringing/calling/trying/connecting: a statement, not an offer,
    not a negation, not a report of a ring that happened. A hold phrase is always a claim."""
    said = _said(sentence)
    if HOLD_PHRASES.search(said):
        return True
    if is_offer(said) or NOT_A_CLAIM.search(said):
        return False
    return any(p.search(said) for p in ACTION_CLAIMS)


# #990 "I need your name to put you through — can you tell me?" was blocked as an offer. Asking for the name or
# the reason is what Jarvis should do then, even when the line says what it's for.
ASKS_FOR_DETAILS = re.compile(r"\b(your name|naam|what it'?s about|what this is about|what the call is about|"
                              r"what is it about|what's this about|what's it about|reason|kis baare)\b")  # #1159


def offers_ring(sentence):
    """True if the sentence offers to ring/try/connect (a question, or an English offer-shaped statement).
    A line that asks for the name or the reason is an ask, not an offer, and passes."""
    said = _said(sentence)
    # A line that asks for (or says it needs) the name or the reason passes, question or not (#990, #1045-1047).
    if not is_offer(said) or NOT_A_CLAIM.search(said) or ASKS_FOR_DETAILS.search(said):
        return False
    return any(p.search(said) for p in RING_OFFERS + ACTION_CLAIMS)


def caller_language():
    """The caller's current language: this turn's verdict with the code overrides, where a short turn ("bye")
    keeps the last clear language (speech_lang), as the goodbye lines do."""
    if not CODE_LANGUAGE:  # option 3: no wait for the judge
        try:
            current_verdict()
        except Exception:
            pass
    return call_state.get("speech_lang") or call_state.get("lang") or "english"


def claim_check(sentence):
    """"claim", "early_offer", "wrong_language", "masculine", or None. Uses THIS turn's judge verdict."""
    if claims_action(sentence):
        return "claim"
    # Live since 2026-11-25 (they used to be eval-only): Hindi words to an English caller, and masculine first
    # person in the female voice. Both were run over every past model line first (44 and 2 hits, all real).
    if has_hindi_words(sentence) and caller_language() == "english":
        return "wrong_language"
    if jarvis_gender() == "female" and MASCULINE_FIRST_PERSON.search(sentence):
        return "masculine"
    if promises_callback(sentence):
        return "promise"
    if offers_ring(sentence):
        try:
            current_verdict()
        except Exception:
            pass
        # Only the reason gates an offer now (2026-11-26, live #1509/#1510): real callers ring Jatin's phone, so
        # wanting him is implied. Requiring caller_wants_jatin too blocked every offer to a caller who gave name
        # and reason but never said "I want Jatin"; the retries were a bare "Thanks, <name>." and the call died.
        if not call_state.get("reason_known"):
            return "early_offer"
    return None


def note_offer(sentence):
    """An offer to ring was just said (by the model or a code line). It stands in for the caller's request:
    try_reach_owner goes on the menu, and the judge reads the caller's next "haan" as agreement to an offer.
    The ring itself still needs their yes (or an explicit request), a reason, and Jatin's yes."""
    record("offer_made", sentence[:200])
    if call_state.get("wants_jatin_turn") is None:
        call_state["wants_jatin_turn"] = call_state["turns"]
        record("implied_request", "offer made after name + reason: counts as the caller wanting Jatin")


# Retry notes: second person, nothing to recite as a line (they must pass leak_check, tested).
CLAIM_RETRY_NOTES = {
    # Each ends by asking for the next question: retries after a block came out as a bare "Thanks, Priya." with no
    # question (#962, #1134, #1457; #1457 lost a ring that way).
    "claim": "You can't ring, call or connect anyone yourself; never say you are. Reply again in one short line, then ask your next question.",
    "early_offer": "Don't offer to try him now. Reply again in one short line, then ask your next question: what's still missing.",
    "wrong_language": "Reply again in English, one short line, then ask your next question.",
    "masculine": "Speak as a woman: karti, rahi, dungi. Reply again in one short line, then ask your next question.",
    "promise": "Never say Jatin will call back or that you'll ask him to; you can note that they want a callback. "
               "Reply again in one short line, then ask your next question.",
}
# If the retry is blocked too and the reason is known (2026-11-26, replaces "I can't put you through right now",
# #1509): offer the next steps. No gendered word. If the reason isn't known, REFUSAL_LINES["reason"] asks for it.
OFFER_FALLBACK_LINES = {
    "english": "Should I try calling him, or take a message or set a reminder?",
    "hinglish": "Main unhe call karne ka try karoon, ya message ya reminder chhod doon?",
}

# Callback promises (live #1509: Jarvis said she'd ask him to call back, "ASAP"). The test is the SUBJECT plus a few
# phrases (Jatin, 2026-11-23 and 2026-11-26): Jarvis may say it will note a callback request, never that Jatin
# will call, nor that it will ask him to. Live guard (claim_check) and eval (run_evals) use this same list.
# Offering a callback ("message or callback?") is allowed, so the word "callback" alone is never a hit.
PROMISE_PATTERNS = [re.compile(p) for p in (
    # Any timing promise, whoever the subject.
    r"\bsoon\b", r"\bjald", r"\bshortly\b", r"\basap\b",
    # English: Jatin / he as the one who will call or get in touch.
    r"\b(he|jatin)\s*('ll|'d| will| would| is going to| can| should)\s+(call|ring|get back|reach out|contact|be in touch|phone)",
    # Hindi: third-person future (lenge/lega/karenge/karega/denge/dega = "he will"), near call/phone/contact.
    # First-person forms ("bol dunga", "bata dunga" = "I will") are Jarvis's own actions and don't match.
    r"\b(call|callback|phone|contact)\b[^.?!]{0,25}\b(kar lenge|kar lega|karenge|karega|lenge|lega|denge|dega)\b",
)]
# Jarvis promising to make him call: a hit only as a statement. As a question it's a callback offer, which is
# allowed ("...or have him call you back?": 16 of 30 past hits). Past statements: 14, all real (#81-#219, #1201, #1509).
ASK_HIM_PATTERNS = [re.compile(p) for p in (
    r"\b(ask|tell|get|have)\s+(him|jatin)\s+(to\s+)?(call|ring|phone|get back|reach out|contact)",  # "I'll ask him to call you back"
    r"\b(call\s*back|callback|call|phone)\s+karne\s+(ko|ke\s+liye)\s+(bol|keh|kah)",  # "call karne ko kahungi" (#1201)
)]


def promises_callback(sentence):
    said = _said(sentence)
    if any(p.search(said) for p in PROMISE_PATTERNS):
        return True
    return not is_question(said) and any(p.search(said) for p in ASK_HIM_PATTERNS)


# Must itself pass leak_check (test_server.py asserts it), or reciting it would trip the guard again.
LEAK_RETRY_NOTE = "Reply again out loud, first person, one short line, then ask your next question."

# If the retry leaks too: a fixed line instead. Keyed on outcome (a connected call must still say so).
# Wording approved by Jatin, 2026-11-23.
LEAK_FALLBACK_LINES = {
    "connected": {"english": "Okay, connecting you now.", "hinglish": "Theek hai, connect kar raha hoon.",
                  "hinglish_female": "Theek hai, connect kar rahi hoon."},
    "other": {"english": "Sorry, can you say that again?", "hinglish": "Sorry, ek baar phir bolna?"},
}

# ---------- what the caller hears: sentence by sentence, and barge-in (2026-11-25) ----------
# The model's reply is streamed. Each complete sentence goes through the leak guards (on everything said so
# far this reply, so a 6-word run across two sentences is still caught) and, if clean, straight to the turn's
# sink (server.py: speech for it starts while the model is still writing the next one). A sentence that fails
# the guard is never spoken; neither is anything after it.
#
# Barge-in: the caller can start talking over Jarvis. The page says how many of this reply's lines played
# (a line counts once at least half of its audio played: "to the nearest sentence"). Every assistant message
# that carries spoken lines is registered in the reply's "parts", so the history can be cut back to exactly
# those lines, whether the turn is still running or already done. Tools are atomic: the cancel flag is read
# BEFORE each tool, never during one, so a save or a ring either went through (kept, logged) or never started.
reply_lock = threading.Lock()  # guards a reply's heard/parts: barge_in runs on a different thread
early_barges = {}              # reply id -> heard, for a barge-in that arrived before its turn began


def new_reply(sink=None, reply_id=None):
    return {"id": reply_id, "sink": sink, "said": [], "parts": [], "cancel": threading.Event(),
            "heard": None, "stream": None, "started": time.time()}


def current_reply():
    if call_state.get("reply") is None:
        call_state["reply"] = new_reply()
    return call_state["reply"]


def begin_reply(sink=None, reply_id=None):
    """A new reply to the caller: one per turn. sink(line) is called with each line as it's cleared to speak."""
    reply = new_reply(sink, reply_id)
    with reply_lock:
        call_state["reply"] = reply
        if reply_id is not None and reply_id in early_barges:
            reply["heard"] = 0  # they started talking before this reply had begun: none of it is heard
            reply["cancel"].set()
            record("barge_in", json.dumps({"turn": call_state.get("turns", 0) + 1, "heard": [], "cut": [], "early": True}))
            early_barges.pop(reply_id)
    return reply


def say(line):
    """Send one line to the caller (via the sink). False if the reply was cancelled: then it isn't sent."""
    reply = current_reply()
    if reply["cancel"].is_set():
        return False
    reply["said"].append(line)
    timing = call_state.setdefault("timing", new_timing())
    if timing.get("first_say") is None:
        timing["first_say"] = int((time.time() - reply["started"]) * 1000) - timing["ring"]
        timing["judge_wait_first"] = timing["judge_wait"]  # judge wait before the first sentence (option 3: ~0)
    if reply["sink"] is not None:
        try:
            reply["sink"](line)
        except Exception as error:  # the call goes on; the line is in the text reply anyway
            record("warn", f"sink failed: {type(error).__name__}")
    return True


def add_spoken(messages, message, lines):
    """Put an assistant message into the history, remembering which spoken lines it holds."""
    with reply_lock:
        messages.append(message)
        reply = current_reply()
        reply["parts"].append((message, list(lines)))
        if reply["heard"] is not None:
            _trim_to_heard(messages, reply)


def _trim_to_heard(messages, reply):
    """Cut this reply's messages back to the lines the caller heard. Idempotent; call with reply_lock held.
    A message with tool calls keeps them (they ran: their results are in the history) and loses only text."""
    keep = reply["heard"]
    for message, lines in reply["parts"]:
        kept, keep = lines[:keep], max(keep - len(lines), 0)
        if message.get("tool_calls"):
            message["content"] = " ".join(kept) or None
        elif kept:
            message["content"] = " ".join(kept)
        else:
            for i, m in enumerate(messages):
                if m is message:
                    del messages[i]
                    break


def barge_in(reply_id, heard):
    """The caller started talking over Jarvis. heard = how many lines of reply reply_id they heard.
    Cancels whatever is still being written or spoken, and cuts the history back to what was heard.
    Safe from any thread; never waits for the turn."""
    with reply_lock:
        reply = call_state.get("reply")
        if reply is None or reply["id"] != reply_id:
            early_barges[reply_id] = heard  # its turn hasn't begun yet (still transcribing): cancel on arrival
            return {"applied": False}
        if reply["heard"] is not None:
            return {"applied": False}  # already cut
        heard = max(0, min(int(heard), len(reply["said"])))
        reply["heard"] = heard
        reply["cancel"].set()
        _trim_to_heard(call_state["messages"], reply)
        record("barge_in", json.dumps({"turn": call_state.get("turns", 0), "heard": reply["said"][:heard],
                                       "cut": reply["said"][heard:]}))
        stream = reply["stream"]
    if stream is not None:  # stop the model mid-reply; the turn's reader sees the cancel flag and stops
        try:
            stream.close()
        except Exception:
            pass
    return {"applied": True, "heard": heard}


def say_line(line):
    """A fixed line from code (server.py's "didn't catch that"): said, logged and put in the history."""
    said = say(line)
    add_spoken(call_state["messages"], {"role": "assistant", "content": line}, [line] if said else [])
    record("jarvis_said", line)
    return line


# A sentence ends at . ! ? or the Hindi full stop, then a space. Under 2 words ("Okay.", "Mr.") it waits
# and joins the next one: no one-word TTS requests, and "Mr. Sharma" stays together.
SENTENCE_END = re.compile(r"[.!?।]+[\"'”’)\]]*\s+")


def split_sentences(buffer):
    """The complete sentences at the front of buffer, and what's left (still being written)."""
    done, start = [], 0
    for match in SENTENCE_END.finditer(buffer):
        piece = buffer[start:match.end()].strip()
        if len(piece.split()) < 2:
            continue
        done.append(piece)
        start = match.end()
    return done, buffer[start:]


def reply_events(response):
    """("text", piece) and ("tool", (index, id, name, arguments piece)) from a streamed reply. A whole,
    non-streamed reply works too (the tests' fakes): it is just one big piece."""
    if hasattr(response, "choices"):
        message = response.choices[0].message
        if message.content:
            yield "text", message.content
        for i, call in enumerate(message.tool_calls or []):
            yield "tool", (i, call.id, call.function.name, call.function.arguments)
        return
    usage_logged = False
    for chunk in response:
        if getattr(chunk, "usage", None) and not usage_logged:
            # Streams report usage on their last chunk (stream_options below). Groq repeats it on a second chunk:
            # log it once, or requests and tokens are counted twice.
            log_usage(MODEL, chunk)
            usage_logged = True
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            yield "text", delta.content
        for call in getattr(delta, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            yield "tool", (call.index, call.id, getattr(function, "name", None), getattr(function, "arguments", None))


STREAM_OPTIONS = {"stream": True, "stream_options": {"include_usage": True}}


def read_reply(response, hint, prompt, hold=False):
    """Read one model reply, speaking each clean sentence as soon as it's complete.
    hold=True (tools were offered): nothing is said until the stream ends, and nothing at all if it ends in a
    tool call. Text before a tool call stays silent, as before streaming (Jatin, 2026-11-25): it could say
    "putting you through now" just before try_reach_owner, which the lock may then refuse.
    Returns emitted (sentences said), problem (None, or (check, text) for the first sentence that failed a
    guard: nothing after it is said), tool_calls, cancelled, failed (the stream broke)."""
    reply = current_reply()
    emitted, buffer, calls, held = [], "", {}, []
    got = {"problem": None, "cancelled": False, "failed": False}

    def offer(sentence):
        if got["problem"] or got["cancelled"]:
            return
        candidate = " ".join(emitted + [sentence])
        leak = "tool_text" if looks_like_a_tool_call(candidate) else leak_check(candidate, hint, prompt)
        claim = None if leak else claim_check(sentence)
        if leak:
            got["problem"] = (leak, candidate)
        elif claim:
            got["problem"] = (claim, sentence)
        elif say(sentence):
            emitted.append(sentence)
            if offers_ring(sentence):
                note_offer(sentence)
        else:
            got["cancelled"] = True

    reply["stream"] = response if not hasattr(response, "choices") else None
    try:
        for kind, value in reply_events(response):
            if reply["cancel"].is_set():
                got["cancelled"] = True
                break
            if kind == "text":
                buffer += value
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    if hold:
                        held.append(sentence)
                    else:
                        offer(sentence)
            else:
                index, call_id, name, arguments = value
                entry = calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                entry["id"] = call_id or entry["id"]
                entry["name"] += name or ""
                entry["arguments"] += arguments or ""
    except Exception as error:
        if reply["cancel"].is_set():
            got["cancelled"] = True  # barge_in closed the stream under us
        else:
            got["failed"] = True
            record("stream_error", f"{type(error).__name__}: {str(error)[:150]}")
    finally:
        reply["stream"] = None
    if reply["cancel"].is_set():
        got["cancelled"] = True
    # The last piece has no space after its full stop. Said only for a text reply: text alongside tool calls
    # is dropped, as before streaming (a first save's follow-up line replaces it). With hold=False (no tools
    # offered) sentences went out as they streamed; "spoke_before_tools" in run_agent can then only fire if a
    # provider sends a tool call without being offered tools.
    rest = held + ([buffer.strip()] if buffer.strip() else [])
    if not calls and not got["cancelled"] and not got["failed"]:
        for sentence in rest:
            offer(sentence)
    got["dropped"] = " ".join(rest) if calls else ""
    got["emitted"] = emitted
    got["tool_calls"] = [SimpleNamespace(id=c["id"], function=SimpleNamespace(name=c["name"], arguments=c["arguments"]))
                         for _, c in sorted(calls.items())]
    return got

# ---------- closing lines (2026-11-23) ----------
# After end_call, code says goodbye, not the model. A goodbye carries no judgment, so it was never the
# model's turn: 3 of the 5 leaks in calls #204-252 were model-written goodbyes after end_call, and the
# turn cost a full model round trip. Fixed lines also get tuned for Sarvam once and stay tuned.
# Full stops, not commas (Jatin's Sarvam finding). No caller name (proposed; Jatin undecided).
# 2026-11-24: English lines reworded to Indian English for Sia 05 (no contractions, "thank you for
# calling"); follow-ups too. Approved by Jatin 2026-11-24.
# Wording approved by Jatin, 2026-11-23. English lines reworded to Indian English (no contractions,
# "thank you for calling") for Sia 05, 2026-11-24. Approved by Jatin 2026-11-24.
CLOSING_LINES = {
    "reminder":       {"english": "Okay, I have noted it for him. Bye.",  "hinglish": "Theek hai, maine note kar liya. Bye."},
    "message":        {"english": "Okay, I will pass it on to him. Bye.",        "hinglish": "Theek hai, main bata dunga. Bye.",
                       "hinglish_female": "Theek hai, main bata dungi. Bye."},
    "nothing_saved":  {"english": "Okay, no problem. Bye, take care.",              "hinglish": "Theek hai. Bye, take care."},
    "spam":           {"english": "No, thank you. Bye.",                "hinglish": "Nahi, thank you. Bye."},
    "abuse":          {"english": "I am ending this call now. Bye.", "hinglish": "Main yeh call end kar raha hoon. Bye.",
                       "hinglish_female": "Main yeh call end kar rahi hoon. Bye."},
    "family_message": {"english": "Okay, I will tell him. Take care.", "hinglish": "Theek hai, main bata dunga. Take care.",
                       "hinglish_female": "Theek hai, main bata dungi. Take care."},
    "family_nothing": {"english": "Okay, take care.",                   "hinglish": "Theek hai. Take care."},
    "family_reminder": {"english": "Okay, I have noted it for him. Take care.", "hinglish": "Theek hai, maine note kar liya. Take care."},
    # After the follow-up was spoken: neutral, because the follow-up already confirmed the save.
    "after_follow_up":        {"english": "Okay, thank you for calling. Take care, bye.", "hinglish": "Theek hai. Bye, take care."},
    "family_after_follow_up": {"english": "Okay, take care.",      "hinglish": "Theek hai. Take care."},
}

# The follow-up (step 4): after the FIRST successful save of a call, code confirms it and asks once,
# in one line. No model request, and the turn ends there. The question is in neither prompt, so the
# model can't recite it. No caller name (Jatin's ruling: names stay in reminders, messages, briefing).
# Wording approved by Jatin, 2026-11-24.
FOLLOW_UP_LINES = {
    "reminder":       {"english": "Okay, I have noted it for him. Anything else you want me to tell him?",
                       "hinglish": "Theek hai, note kar liya. Aur kuch batana hai unhe?"},
    "message":        {"english": "Okay, I will pass it on to him. Anything else you want me to tell him?",
                       "hinglish": "Theek hai, main bata dunga. Aur kuch batana hai unhe?",
                       "hinglish_female": "Theek hai, main bata dungi. Aur kuch batana hai unhe?"},
    "family_message": {"english": "Okay, I will tell him. Anything else you want me to tell him?",
                       "hinglish": "Theek hai, main bata dunga. Aur kuch batana hai unhe?",
                       "hinglish_female": "Theek hai, main bata dungi. Aur kuch batana hai unhe?"},
}


def jarvis_gender():
    """The active voice's gender (voice.py). Hinglish first person follows it: karta/karti, dunga/dungi."""
    try:
        return voice.gender()
    except Exception:
        return "male"


def pick_line(lines, lang):
    """A fixed line in this language, in the grammar of the active voice's gender. Lines with no
    gendered word have no "hinglish_female" entry and are the same for both."""
    if lang == "hinglish" and jarvis_gender() == "female":
        return lines.get("hinglish_female", lines["hinglish"])
    return lines.get(lang, lines["english"])


def all_fixed_lines():
    """Every line code can speak, by language, in the active voice's gender: the cache pre-renders
    these (voice.py) so they play with no TTS delay. server.py adds its own (hold line, didn't-catch)."""
    lines = {"english": set(), "hinglish": {OPENER} | {o["hinglish"] for o in FAMILY_OPENERS.values()}}  # openers: Hinglish
    tables = [LEAK_FALLBACK_LINES, CLOSING_LINES, FOLLOW_UP_LINES, {"x": LINES_NOTE_SAVED}, {"x": LINES_NOTE_NOT_SAVED},
              {"x": DECLINE_LINES}, FAMILY_RING_LINES, {"x": CONNECTED_LINES}, {"x": ADDED_LINES}, {"x": DUPLICATE_LINES},
              {"x": OFFER_FALLBACK_LINES}, {"x": REASON_AGAIN_LINES}] + [REFUSAL_LINES]
    for table in tables:
        for entry in table.values():
            for lang in ("english", "hinglish"):
                lines[lang].add(pick_line(entry, lang))
    for lang in ("english", "hinglish"):  # the two filler lists differ in length: each language gets its own list
        lines[lang].update(FILLER_LINES[lang])
    return lines


def fixed_line_lang():
    """Language for a fixed line, and for how a reply is spoken: the caller's latest substantive turn.
    Not simply the latest verdict: the judge read a Hinglish caller's final "bye" as English (#242)."""
    if not CODE_LANGUAGE:  # option 3: the language is already known, so no wait for the judge
        try:
            current_verdict()  # sets the voice on turn 1 if the call ends on its first turn
        except Exception:
            pass  # nothing may stop the caller hearing a line
    return call_state.get("speech_lang") or call_state.get("lang") or "english"


def closing_key():
    """Which closing line. Spam and abuse stay flat even after a save. Otherwise, if the follow-up was
    said it already confirmed the save, so the goodbye is neutral. The outcome lines are for a save
    with no follow-up said, which code now prevents; they stay as the safe fallback."""
    if call_state["family"]:
        if call_state["follow_up_spoken"]:
            return "family_after_follow_up"
        return {"message": "family_message", "reminder": "family_reminder"}.get(call_state["outcome"], "family_nothing")
    if call_state.get("end_kind") == "spam":
        return "spam"
    if call_state.get("end_kind") == "abuse":
        return "abuse"
    if call_state["follow_up_spoken"]:
        return "after_follow_up"
    if call_state["outcome"] in ("reminder", "message"):
        return call_state["outcome"]
    return "nothing_saved"


def say_fixed_line(messages, table, key, event_note):
    lines = table[key]
    line = pick_line(lines, fixed_line_lang())
    said = say(line)
    add_spoken(messages, {"role": "assistant", "content": line}, [line] if said else [])
    record("jarvis_said", line)
    record("fixed_line", f"{event_note}: {key}")
    return line

# After a ring Jatin declined or didn't answer: code says this, never the model (live call 2026-11-25, Jatin's
# wording). No gendered word in either line, so no "hinglish_female" entry: the same line for any voice.
DECLINE_LINES = {
    "english": "Jatin couldn't take the call right now. Should I take a message or set a reminder for him?",
    "hinglish": "Jatin abhi call nahi le paaye. Message chhod doon ya reminder set kar doon?",
}

DECLINE_UNHEARD_RESULT = ("Not connected. Tell them he couldn't take it right now, then offer a message or a reminder "
                          "for him. Save nothing until they answer.")

# After a refused ring (Jatin's wording, 2026-11-25). No gendered word: the same line for any voice.
REFUSAL_LINES = {
    "reason": {"english": "Could you tell me what it's about first?",
               "hinglish": "Pehle bata dijiye, kis baare mein baat karni hai?"},
    "name": {"english": "Could I have your name first?", "hinglish": "Pehle aapka naam bata dijiye?"},
    "not_agreed": {"english": "Should I try calling him now, or save a reminder for him?",
                   "hinglish": "Kya main unhe abhi call karne ka try karoon, ya unke liye reminder set kar doon?"},
    "tried": {"english": "I've already tried him once. Should I take a message or set a reminder for him?",
              "hinglish": "Jatin ko abhi try kiya tha. Message chhod doon ya reminder set kar doon?"},
    # Family: another ring refused (after a decline, or after the one re-ring).
    "family_declined": {"english": "I can't try him again right now. Should I remind him to call you back, or take a message?",
                        "hinglish": "Abhi unhe dobara try nahi kar sakta. Remind kar doon, ya koi message?",
                        "hinglish_female": "Abhi unhe dobara try nahi kar sakti. Remind kar doon, ya koi message?"},
    "family_tried": {"english": "I've already tried him twice. Should I remind him to call you back, or take a message?",
                     "hinglish": "Unhe do baar try kar liya hai. Remind kar doon, ya koi message?"},
}
# #1625: "Could you tell me what it's about first?" on two turns in a row sounded stuck. If code said it on the
# previous turn, it says this instead (Jatin's wording, 2026-11-26); the turn after, the usual line again.
REASON_AGAIN_LINES = {"english": "Just tell me what the call's about?",
                      "hinglish": "Bas itna bata dijiye, call kis baare mein hai?"}


def reason_ask_lines():
    """The reason ask code says now: REFUSAL_LINES["reason"], or REASON_AGAIN_LINES if that was said last turn."""
    again = call_state.get("reason_ask_turn") == call_state["turns"] - 1
    call_state["reason_ask_turn"] = None if again else call_state["turns"]
    return REASON_AGAIN_LINES if again else REFUSAL_LINES["reason"]


# What the model reads: the refusal, and that the caller has just been asked (removed if a barge-in cut the line).
# Not a refusal line: the model may call try_reach_owner again in this turn with this fixed (#1404).
REASON_ARGUMENT_MISSING = ("ERROR: reason missing from this call. Pass what they said it's about as reason, or, if they "
                           "said there's no particular reason, set no_reason_given to true. Then call try_reach_owner again.")

REFUSAL_RESULTS = {
    "reason": "ERROR: no reason yet; they've just been asked what it's about.",
    "name": "ERROR: no name yet; they've just been asked for it.",
    "not_agreed": "ERROR: not agreed yet; they've just been asked whether to try him or save a reminder.",
    "tried": "ERROR: already tried once; they've just been offered a message or a reminder.",
    "family_declined": "ERROR: he can't be tried again; they've just been offered a reminder or a message.",
    "family_tried": "ERROR: already tried twice; they've just been offered a reminder or a message.",
}
# Family re-ring (Jatin, 2026-11-26). After the one re-ring fails, and when another ring is refused. Never
# "declined" to family. Female forms where a word is gendered (pick_line).
FAMILY_RING_LINES = {
    "family_no_pickup": {"english": "He didn't pick up this time either. Should I remind him to call you back, or take a message?",
                         "hinglish": "Jatin ne is baar bhi phone nahi uthaya. Remind kar doon, ya koi message?"},
    "family_declined": {"english": "Jatin couldn't take the call right now. Should I remind him to call you back, or take a message?",
                        "hinglish": "Jatin abhi call nahi le paaye. Remind kar doon, ya koi message?"},
}

# Three more moments where the model always said the same thing (calls #679-849), now said by code
# (Jatin, 2026-11-25). The model's text in those turns is dropped.
CONNECTED_LINES = {"english": "Connecting you now.", "hinglish": "Connect kar raha hoon.",
                   "hinglish_female": "Connect kar rahi hoon."}
ADDED_LINES = {"english": "Got it, I'll pass that on too.", "hinglish": "Theek hai, yeh bhi bata dunga.",
               "hinglish_female": "Theek hai, yeh bhi bata dungi."}
DUPLICATE_LINES = {"english": "That's already with him.", "hinglish": "Woh already note ho gaya hai."}
# What those tool results tell the model the caller heard. Removed if a barge-in cut the line first.
HEARD_CLAIMS = (" They've heard it confirmed.", " They've heard it's already with him.")

# A filler for slow turns (2026-11-25): the page plays it when no audio has started 1.5 s after the caller let go,
# at most once per turn, from the cache. Not in the model's history (like the hold line). No gendered word.
# 2026-11-26 (live #1509 at 2.0 s: a filler on all 7 turns, "hmm" every third): picked at random, never the same one
# twice in a row, never on two turns in a row, and skipped at random about a third of the time. With a filler due on
# most turns that plays one on about 40% of them (after a skip: 2/3; after a filler: 0). No word the model often starts
# a reply with (Jatin: "Got it... Got it, Rahul", "Theek hai... Theek hai Rohit"); run_evals counts any that slip.
FILLER_LINES = {"english": ["Hmm...", "One second...", "Right...", "Alright...", "Let me see..."],
                "hinglish": ["Achha...", "Hmm...", "Ek second...", "Ek minute..."]}
FILLER_SKIP_SHARE = 1 / 3
filler_random = random.Random()  # tests seed or replace it


# Fillers by situation (Jatin, 2026-11-26): "Let me see..." before "Okay, take care" sounded wrong. Code picks the
# group from what it knows the moment the filler fires (Jarvis's last line and the caller's line); never the judge,
# which may not have answered within 2 s. First match wins:
#   1. Jarvis just asked "anything else?" (a follow-up line)          -> no filler (they're almost always saying bye)
#   2. the caller asked to be put through, or said yes to an offer      -> working
#   3. the caller asked Jarvis something                                -> working
#   4. the caller answered a question / 5. anything else                -> acknowledgement
FILLER_GROUPS = {"ack": {"english": ["Hmm...", "Right...", "Alright..."], "hinglish": ["Achha...", "Hmm..."]},
                 "working": {"english": ["One second...", "Let me see..."], "hinglish": ["Ek second...", "Ek minute..."]}}
# Shared with run_evals.bad_rings: the caller explicitly asking to be put through or tried.
EXPLICIT_RING_REQUEST = re.compile(
    r"\b(connect me|put me through|try him|try again|ring him|call him|try (jatin|karo|kar lo|kar do|kijiye)|"
    r"connect (karo|kar do|kijiye)|baat karwa|phone pe lao)\b", re.I)
YES_WORDS = {"haan", "han", "haa", "ha", "yes", "yeah", "yep", "ok", "okay", "sure", "theek", "thik", "try", "please", "ji",
             "karo", "kijiye", "go"}
NO_WORDS = {"no", "nahi", "nahin", "nah", "mat", "don't", "dont", "not", "rehne"}
QUESTION_WORDS = {"what", "where", "when", "who", "why", "how", "can", "is", "are", "kya", "kab", "kahan", "kaun", "kyun",
                  "kyon", "kaise"}


def filler_kind(last_jarvis, caller_line):
    """"none", "working" or "ack" for a filler, from Jarvis's last line and the caller's line (rules above)."""
    last = (last_jarvis or "").strip()
    if last in {pick_line(lines, lang) for lines in FOLLOW_UP_LINES.values() for lang in ("english", "hinglish")} \
            or last in {lines[k] for lines in FOLLOW_UP_LINES.values() for k in lines}:
        return "none"
    said = (caller_line or "").lower().replace("’", "'")
    words = re.findall(r"[a-z']+", said)
    if EXPLICIT_RING_REQUEST.search(said):
        return "working"
    offered = any(offers_ring(s) for s in split_sentences(last + " ")[0] or [last])
    if offered and words and set(words) & YES_WORDS and not set(words) & NO_WORDS:
        return "working"
    if said.rstrip().endswith("?") or (words and words[0] in QUESTION_WORDS):
        return "working"
    return "ack"


def last_lines():
    """(Jarvis's last spoken line before the caller's latest turn, the caller's latest line) from the history."""
    messages = call_state.get("messages") or []
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            before = [m for m in messages[:i] if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")]
            return (before[-1]["content"] if before else ""), messages[i]["content"]
    return "", ""


def pick_filler(state, lang, turn, rng, kind="ack"):
    """The filler to play on this turn, or None (a skip, or "none" for the situation). Updates state
    (last_filler, last_filler_turn). Shared by the live server (next_filler) and run_evals' simulation."""
    if kind == "none":
        return None
    last_turn = state.get("last_filler_turn")
    if last_turn is not None and turn - last_turn <= 1:
        return None  # one on the previous turn (or already this turn): skip
    if rng.random() < FILLER_SKIP_SHARE:
        return None
    group = FILLER_GROUPS[kind]["hinglish" if lang == "hinglish" else "english"]
    options = [o for o in group if o != state.get("last_filler")] or group
    choice = rng.choice(options)
    state["last_filler"], state["last_filler_turn"] = choice, turn
    return choice


def next_filler(lang):
    """(filler or None, situation). Called only when the page asks (a slow turn)."""
    kind = filler_kind(*last_lines())
    return pick_filler(call_state, lang, call_state.get("turns", 0), filler_random, kind), kind

# Two versions of the line: Jarvis only says "I've noted it" if the note really saved.
LINES_NOTE_SAVED = {
    "english": "Sorry, something's gone wrong on my end. I've noted that you called and I'll pass it to Jatin.",
    "hinglish": "Sorry, mere side kuch gadbad ho gayi. Maine note kar liya hai ki aapne call kiya, Jatin ko bata dunga.",
    "hinglish_female": "Sorry, mere side kuch gadbad ho gayi. Maine note kar liya hai ki aapne call kiya, Jatin ko bata dungi.",
}
LINES_NOTE_NOT_SAVED = {
    "english": "Sorry, something's gone wrong on my end and I can't take your details right now. Please try Jatin again in a bit.",
    "hinglish": "Sorry, mere side kuch gadbad ho gayi hai, abhi main aapki details nahi le pa raha. Thodi der baad Jatin ko dobara try kar lijiye.",
    "hinglish_female": "Sorry, mere side kuch gadbad ho gayi hai, abhi main aapki details nahi le pa rahi. Thodi der baad Jatin ko dobara try kar lijiye.",
}


def save_failed_call_note():
    """Save whatever we know about the caller, marked INCOMPLETE.
    Returns True if it saved, False if not. Never raises."""
    try:
        who = call_state.get("heard_name") or call_state.get("caller_name")
        if not who:
            who = f"Unknown caller (caller ID {call_state['caller_id'] or 'unknown'})"
        reason = call_state.get("heard_reason")
        if reason:
            content = f"INCOMPLETE - system failed mid-call. What we had: {reason}"
        else:
            content = "INCOMPLETE - caller reached Jarvis but the system failed mid-call"
        db.save_reminder(call_state["call_id"], "failed_call", who, content)
        return True
    except Exception as error:  # anything at all: the caller must still hear a line
        try:
            record("warn", f"could not save failed-call note: {type(error).__name__}")
        except Exception:
            pass
        return False


def end_call_with_note(messages, outcome):
    """Something failed that the conversation can't recover from (provider down,
    or the agent loop got stuck): save a note, say a fixed line, end the call.
    The line says "noted" only if the note really saved. Nothing here can stop
    the caller hearing a line."""
    if not CODE_LANGUAGE:
        try:
            current_verdict()  # the caller's language, for the line below
        except Exception:
            pass  # nothing may stop the caller hearing a line
    saved = save_failed_call_note()
    lines = LINES_NOTE_SAVED if saved else LINES_NOTE_NOT_SAVED
    # The call's voice language, like every other fixed line: the line is pre-rendered in that voice.
    line = pick_line(lines, fixed_line_lang())
    try:
        said = say(line)
    except Exception:
        said = False
    try:
        add_spoken(messages, {"role": "assistant", "content": line}, [line] if said else [])
    except Exception:
        messages.append({"role": "assistant", "content": line})
    try:
        record("jarvis_said", line)
    except Exception:
        pass
    call_state["ended"] = True
    call_state["outcome"] = outcome
    return line


def run_agent(messages):
    reply = current_reply()
    latest = messages[-1]["content"]
    tools_allowed = not is_greeting_only(latest)
    if call_state["family"]:
        active_tools = family_tools
        allowed_names = {"take_message", "add_reminder", "try_reach_owner", "end_call"}
    else:
        active_tools = tools
        allowed_names = set(TOOL_FUNCTIONS)
        # try_reach_owner isn't on the menu until caller_wants_jatin has fired (#884: it rang first and asked later;
        # #912). Since 2026-11-26 that includes earlier in THIS turn: the menu is rebuilt every step, so the model
        # can call caller_wants_jatin, then try_reach_owner in its next step (#1176, #1399: a request and a "yes" in
        # one line cost a turn). The locks still need an offer, their yes and a reason before anything rings.
    ran_this_turn = []
    leak_retried = False  # one retry per turn; a second leak gets a fixed line
    claim_retried = False  # the same for the claim guard

    def said_this_turn():
        return " ".join(reply["said"])

    for step in range(MAX_STEPS):
        if reply["cancel"].is_set():
            record("cancelled", f"barge-in before model request {step + 1}: nothing more is said or done")
            return said_this_turn()
        call_messages = messages + [{"role": "system", "content": state_hint()}]

        if not call_state["family"]:
            if call_state["wants_jatin_turn"] is None:
                active_tools = [t for t in tools if t["function"]["name"] != "try_reach_owner"]
                allowed_names = set(TOOL_FUNCTIONS) - {"try_reach_owner"}
            else:
                active_tools, allowed_names = tools, set(TOOL_FUNCTIONS)
        options = {"model": MODEL, "messages": call_messages, **MAIN_OPTIONS, **STREAM_OPTIONS}
        if tools_allowed:
            options["tools"] = active_tools

        started = time.time()
        try:
            response = call_model(**options)
        except BadRequestError:
            print("   [warn] bad tool call from the model, retrying without tools")
            record("warn", "bad tool call, retrying without tools")
            try:
                response = call_model(model=MODEL, messages=call_messages, **MAIN_OPTIONS, **STREAM_OPTIONS)
            except BadRequestError:
                print("   [warn] retry failed too, using the fixed fallback line")
                record("warn", "retry failed, used fallback line")
                add_time("model", started)
                say_line("Sorry, didn't catch that. Can you say it again?")
                return said_this_turn()

        if response is None:
            add_time("model", started)
            end_call_with_note(messages, "provider_unavailable")
            return said_this_turn()
        got = read_reply(response, call_messages[-1]["content"], messages[0]["content"], hold="tools" in options)
        add_time("model", started)  # includes any retries and the whole stream: time the caller waited
        emitted = got["emitted"]

        if got["cancelled"] or got["failed"]:
            # Barge-in (or a broken stream): what was already said stays in the history (cut to what was
            # heard); a tool call that hadn't run never runs.
            if emitted:
                add_spoken(messages, {"role": "assistant", "content": " ".join(emitted)}, emitted)
                record("jarvis_said", " ".join(emitted))
            dropped = [c.function.name for c in got["tool_calls"]]
            if got["cancelled"]:
                record("cancelled", f"barge-in during model reply {step + 1}"
                       + (f"; tool calls not run: {', '.join(dropped)}" if dropped else ""))
                return said_this_turn()
            if reply["said"]:
                return said_this_turn()  # the caller has a reply; don't add a second one
            continue  # nothing said yet: ask the model again (counts as a step)

        if not got["tool_calls"]:
            if got["problem"] is None:
                if not emitted:  # an empty reply
                    say_line("Sorry, can you say that again?")
                    return said_this_turn()
                text = " ".join(emitted)
                add_spoken(messages, {"role": "assistant", "content": text}, emitted)
                record("jarvis_said", text)
                return said_this_turn()
            check_name, bad = got["problem"]
            if emitted:
                # Part of the reply was already said: stop at the last clean sentence. No retry, since a
                # second reply after a half-said one would be two answers to one turn.
                if check_name == "tool_text":
                    record("warn", f"model wrote a tool call as text: {bad[:120]}")
                elif check_name in CLAIM_RETRY_NOTES:
                    record("claim_guard", f"{check_name}: {bad[:200]}")
                else:
                    record("leak_guard", f"{check_name}: {bad[:200]}")
                record("leak_stop", f"said the {len(emitted)} clean sentence(s) before it, nothing after")
                if check_name == "early_offer" and not any(is_question(s) for s in emitted):
                    # #1565-1567: "Sorry, I can't chat about the match." then a blocked offer (no reason yet): the
                    # steer-back question was lost. Ask for the reason instead (Jatin's wording, 2026-11-26).
                    line = pick_line(reason_ask_lines(), fixed_line_lang())
                    if say(line):
                        emitted.append(line)
                        record("fixed_line", "early_offer after a clean sentence: reason")
                text = " ".join(emitted)
                add_spoken(messages, {"role": "assistant", "content": text}, emitted)
                record("jarvis_said", text)
                return said_this_turn()
            if check_name == "tool_text":
                record("warn", f"model wrote a tool call as text: {bad[:120]}")
                messages.append({
                    "role": "system",
                    "content": "Never write a tool call as text. Use the real tool, then reply in words only.",
                })
                continue
            if check_name in CLAIM_RETRY_NOTES:
                record("claim_guard", f"{check_name}: {bad[:200]}")  # never said
                if not claim_retried:
                    claim_retried = True
                    messages.append({"role": "system", "content": CLAIM_RETRY_NOTES[check_name]})
                    continue
                if check_name in ("claim", "early_offer", "promise"):
                    if call_state["reason_known"]:
                        # Offer the next steps instead of "I can't put you through right now" (#1509).
                        line = say_fixed_line(messages, {"offer": OFFER_FALLBACK_LINES}, "offer", f"{check_name} retry blocked too")
                        note_offer(line)
                    else:  # no reason yet: an offer would only be refused, so ask for it
                        say_fixed_line(messages, {"reason": reason_ask_lines()}, "reason", f"{check_name} retry blocked too")
                else:  # language or gender retried and failed again: the neutral fixed line, in the caller's language
                    say_fixed_line(messages, LEAK_FALLBACK_LINES, "other", f"{check_name} retry blocked too")
                return said_this_turn()
            record("leak_guard", f"{check_name}: {bad[:200]}")  # the line is never said
            if not leak_retried:
                leak_retried = True
                messages.append({"role": "system", "content": LEAK_RETRY_NOTE})
                continue
            key = "connected" if call_state["connected"] else "other"
            say_fixed_line(messages, LEAK_FALLBACK_LINES, key, "retry leaked too")
            return said_this_turn()

        tool_calls = got["tool_calls"]
        if emitted:
            record("spoke_before_tools", " ".join(emitted))
            record("jarvis_said", " ".join(emitted))
        message = {"role": "assistant", "content": " ".join(emitted) or None,
                   "tool_calls": [{"id": c.id, "type": "function",
                                   "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                  for c in tool_calls]}
        add_spoken(messages, message, emitted)
        call_ended_here = False
        # Every first save gets the follow-up: end_call can't ride along with it, in either order.
        # (A second save + end_call is allowed: if their line already said bye, ending is right, and code
        # can't tell without a word list. "Wait for their bye" after a second save is prompt-only.)
        first_save_in_batch = not call_state["follow_up_spoken"] and any(
            c.function.name in ("add_reminder", "take_message") for c in tool_calls)

        for position, call in enumerate(tool_calls):
            if reply["cancel"].is_set():
                # Barge-in between tools: the ones not started never happen, and their calls leave the
                # history too (every tool call in the history needs a result).
                skipped = tool_calls[position:]
                with reply_lock:
                    message["tool_calls"] = message["tool_calls"][:position]
                    if not message["tool_calls"]:
                        del message["tool_calls"]
                        if not message.get("content"):
                            messages[:] = [m for m in messages if m is not message]
                record("cancelled", "barge-in before tools: not run: " + ", ".join(c.function.name for c in skipped))
                break
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = None
            blockers = [t for t in ran_this_turn if t in ("caller_wants_jatin", "try_reach_owner")]

            record("tool_called", f"{name} {json.dumps(args) if args is not None else call.function.arguments}")
            note_what_we_heard(args)

            if not isinstance(args, dict):
                result = "ERROR: wrong arguments. Retry with the correct ones."
            elif name == "try_reach_owner" and name not in allowed_names and not call_state["family"]:
                result = "ERROR: use caller_wants_jatin first, then wait for their answer."
            elif name not in allowed_names:
                result = "ERROR: not available. Reply in plain words."
            elif name in ran_this_turn:
                result = "Already done this turn. Don't repeat it; reply to the caller now."
            elif name == "end_call" and blockers:
                result = "ERROR: can't end the call in the same turn as that. Reply and wait for the caller."
            elif name == "end_call" and first_save_in_batch:
                result = "ERROR: can't end the call in the same turn as the first save. Wait for their answer."
            else:
                function = TOOL_FUNCTIONS[name]
                timing = call_state.setdefault("timing", new_timing())
                started, ring_before, wait_before = time.time(), timing["ring"], timing["judge_wait"]
                try:
                    result = function(**args)
                except TypeError:
                    result = "ERROR: wrong arguments. Retry with the correct ones."
                add_time("tools", started)
                # The ring and the wait for the judge happen inside some tools; each is counted on its own.
                timing["tools"] -= (timing["ring"] - ring_before) + (timing["judge_wait"] - wait_before)

            if result != REASON_ARGUMENT_MISSING:  # an argument fix may be retried in the same turn (#1404)
                ran_this_turn.append(name)
            # What the agent DID vs what it TRIED: a refusal from the code ("ERROR..." or
            # "Already done...") means the tool did not go through.
            if result.startswith("ERROR") or result.startswith("Already done"):
                call_state["tools_blocked"].append(name)
            else:
                call_state["tools_used"].append(name)
                if reply["cancel"].is_set():
                    record("kept_after_barge_in", f"{name} went through before the barge-in: kept")
            if call_state["verbose"]:
                print("   [tool]", name, "->", result)
            record("tool_result", f"{name} -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if name == "end_call" and result.startswith("Call ended"):
                call_ended_here = True

        # end_call went through: code says goodbye. No model turn after end_call.
        if call_ended_here:
            call_state["follow_up_due"] = None  # ended in the same turn as the save: no follow-up
            say_fixed_line(messages, CLOSING_LINES, closing_key(), "closing line")
            return said_this_turn()
        if reply["cancel"].is_set():
            if call_state.get("decline_due"):
                # They talked over Jarvis before the decline line: the result must not say they heard it.
                call_state["decline_due"] = False
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "tool" and m["content"].startswith("Not connected."):
                        m["content"] = DECLINE_UNHEARD_RESULT
                        break
            # They're already talking: no follow-up and no reply now. follow_up_spoken stays False, so the
            # next save still gets the follow-up, and a bye gets the outcome's closing line.
            call_state["follow_up_due"] = None
            call_state["added_due"] = call_state["duplicate_due"] = False
            call_state["refusal_due"] = None
            # The code line those results promise was never said: take the promise out of them.
            batch_start = next((i for i, m in enumerate(messages) if m is message), len(messages))
            for m in messages[batch_start:]:  # this batch's results only: earlier lines were heard
                if isinstance(m, dict) and m.get("role") == "tool":
                    for heard in HEARD_CLAIMS:
                        m["content"] = m["content"].replace(heard, "").strip()
                    m["content"] = re.sub(r"; they've just been [^.]*\.", ".", m["content"])
            record("cancelled", "barge-in after tools: no reply written")
            return said_this_turn()
        # Lines code says after the batch, no model request. At most one; the model's text is dropped.
        # Order: connected, declined, first save (the follow-up, below), a later save, a duplicate.
        refusal = call_state.get("refusal_due")
        call_state["refusal_due"] = None
        due = [("connected", call_state["connected"], CONNECTED_LINES),
               ("decline", call_state.get("decline_due"),
                DECLINE_LINES if call_state.get("decline_due") is True else FAMILY_RING_LINES.get(call_state.get("decline_due"))),
               (f"refusal_{refusal}", bool(refusal), reason_ask_lines() if refusal == "reason" else REFUSAL_LINES.get(refusal))]
        later = [("added", call_state.get("added_due"), ADDED_LINES),
                 ("duplicate", call_state.get("duplicate_due"), DUPLICATE_LINES)]
        call_state["decline_due"] = False
        for name, is_due, lines in due:
            if is_due:
                call_state["added_due"] = call_state["duplicate_due"] = False
                if got["dropped"]:
                    record("dropped_text", f"replaced by the {name} line: {got['dropped'][:200]}")
                say_fixed_line(messages, {name: lines}, name, f"{name} line")
                return said_this_turn()
        # First save of the call went through: code confirms it and asks once. No model request; any
        # text the model wrote alongside the save is dropped, so "noted" is never said twice.
        if call_state["follow_up_due"] and not call_state["ended"]:
            kind = call_state["follow_up_due"]
            call_state["follow_up_due"] = None
            call_state["follow_up_spoken"] = True
            call_state["added_due"] = call_state["duplicate_due"] = False
            if got["dropped"]:
                record("dropped_text", f"said alongside the save, replaced by the follow-up: {got['dropped'][:200]}")
            say_fixed_line(messages, FOLLOW_UP_LINES, kind, "follow-up")
            return said_this_turn()
        for name, is_due, lines in later:
            if is_due:
                call_state["added_due"] = call_state["duplicate_due"] = False
                if got["dropped"]:
                    record("dropped_text", f"replaced by the {name} line: {got['dropped'][:200]}")
                say_fixed_line(messages, {name: lines}, name, f"{name} line")
                return said_this_turn()

    # MAX_STEPS model calls in one turn and still no reply for the caller: the loop is stuck.
    call_state["stuck"] = True
    record("agent_stuck", f"no reply after {MAX_STEPS} steps")
    end_call_with_note(messages, "agent_stuck")
    return said_this_turn()


# ---------- input, from a person or from a script ----------

def next_caller_line():
    queue = call_state["caller_queue"]
    if queue is None:
        return input("Caller: ")
    if not queue:
        return "quit"
    line = queue.pop(0)
    if call_state["verbose"]:
        print("Caller:", line)
    return line


def next_owner_answer(prompt):
    queue = call_state["owner_queue"]
    if queue is None:
        return input(prompt)
    answer = queue.pop(0) if queue else ""
    if call_state["verbose"]:
        print(prompt + answer)
    return answer


# ---------- one call, start to finish ----------

def reset_state(caller_id, is_family, caller_queue, owner_queue, verbose):
    call_state.clear()
    call_state.update({
        "ended": False,
        "turns": 0,
        "wants_jatin_turn": None,
        "caller_agreed": False,
        "reason_known": False,
        "rings": 0,
        "family": is_family,
        "caller_id": caller_id,
        "lang": "english",
        "speech_lang": None,   # language of the caller's latest substantive turn: fixed lines follow it
        "call_id": None,
        "caller_name": None,
        "outcome": None,
        "connected": False,
        "caller_queue": caller_queue,
        "owner_queue": owner_queue,
        "verbose": verbose,
        "tools_used": [],      # tool calls that went through
        "tools_blocked": [],   # tool calls the code refused
        "provider_down": False,
        "stuck": False,
        "heard_name": None,
        "heard_reason": None,
        "follow_up_due": None,     # "reminder" / "message" / "family_message" once the first save succeeds
        "follow_up_spoken": False, # the code-spoken "anything else?" line: at most once per call
        "saved_messages": [],      # take_message notes saved on this call (duplicate check)
        "decline_due": False,      # a ring just failed: code says DECLINE_LINES after the tool batch
        "refusal_due": None,       # a ring was refused: code says REFUSAL_LINES[key] after the tool batch
        "added_due": False,        # a save after the follow-up: code says ADDED_LINES
        "duplicate_due": False,    # a duplicate save was refused: code says DUPLICATE_LINES
        "reply": new_reply(),      # what the caller is hearing now (streaming, barge-in)
    })
    early_barges.clear()


def start_call(caller_id="", caller_lines=None, owner_answers=None, verbose=True):
    """Set up a new call. Returns Jarvis's opening line, or None if the call
    already ended (a family call that Jatin picked up straight away).
    owner_answers: None = ask in the terminal, a list = scripted answers (evals),
    a function = the web page's phone panel (it gets the briefing, returns the answer)."""
    db.setup()
    is_family = caller_id in FAMILY_IDS
    reset_state(caller_id, is_family, caller_lines, owner_answers, verbose)

    call_state["call_id"] = db.start_call(caller_id or "unknown", is_family)
    if verbose:
        print(f"[System] Call #{call_state['call_id']} started.")

    system_text = prompt_for_voice(SYSTEM_PROMPT)
    # Fixed in code, so every caller hears it exactly. "AI assistant" is a disclosure, not style: keep it.
    opener = OPENER

    if is_family:
        if verbose:
            print("[System] Family call: ringing Jatin straight away, no screening.")
        record("family_call", f"caller ID {caller_id} matched family list")
        name = family_shown()  # the briefing shows on the page's phone panel
        if ask_owner(f"Family call: {name} (caller ID {caller_id})" if name not in ("Family member", "Family")
                     else f"Family call (caller ID {caller_id})"):
            if verbose:
                print("[System] Connected.")
            call_state["ended"] = True
            call_state["connected"] = True
            call_state["outcome"] = "connected"
        else:
            call_state["rings"] = 1
            system_text = prompt_for_voice(FAMILY_PROMPT)
            opener = FAMILY_OPENERS[call_state["last_ring"]]["hinglish"]  # declined / no_pickup

    # The conversation history lives in call_state, so each turn can be handled separately
    # (one function call per turn, which is what the web server will need).
    call_state["messages"] = [
        {"role": "system", "content": system_text},
        {"role": "assistant", "content": opener},
    ]

    if call_state["ended"]:
        return None
    if verbose:
        print("Jarvis is answering. Type 'quit' to hang up.")
        print("Jarvis:", opener)
    record("jarvis_said", opener)
    return opener


def handle_turn(user_text, sink=None, reply_id=None):
    """One caller turn: log what they said, run the judge, run the agent.
    sink(line): called with each line as soon as it's cleared to speak (server.py streams it).
    reply_id: the page's id for this reply, so a barge-in names the reply it cut.
    Returns everything Jarvis said this turn. Check call_state["ended"] afterwards."""
    begin_reply(sink, reply_id)
    messages = call_state["messages"]
    messages.append({"role": "user", "content": user_text})
    record("caller_said", user_text)
    call_state["turns"] += 1
    call_state["timing"] = new_timing()
    if CODE_LANGUAGE:
        set_code_language(user_text)
    start_judge(messages)   # in the background, alongside the model
    reply = run_agent(messages)
    current_verdict()       # make sure it's applied before the turn ends (voice, next turn's facts)
    if call_state["verbose"]:
        print("Jarvis:", reply)
    return reply


def caller_silent():
    """They said nothing after Jarvis's last line (meant for after the follow-up). End the call with
    the closing line: neutral if the follow-up was said. The outcome stays whatever was saved.
    Returns the line, or None if the call had already ended."""
    if call_state["ended"]:
        return None
    record("caller_silent", "no answer: ending the call")
    call_state["ended"] = True
    call_state["end_reason"] = "caller went silent"
    return say_fixed_line(call_state["messages"], CLOSING_LINES, closing_key(), "closing line")


def finish_call(hung_up=False):
    """Close the call in the database and return a summary of what happened."""
    if hung_up:
        record("caller_hung_up", "caller hung up")
    if call_state["outcome"] is None:
        call_state["outcome"] = "abandoned"

    db.end_call(
        call_state["call_id"],
        call_state["outcome"],
        call_state["connected"],
        call_state["caller_name"],
    )
    record("call_ended", f"outcome={call_state['outcome']} connected={call_state['connected']}")
    if call_state["verbose"]:
        print(f"[System] Call #{call_state['call_id']} ended. Outcome: {call_state['outcome']}")

    return {
        "call_id": call_state["call_id"],
        "outcome": call_state["outcome"],
        "connected": call_state["connected"],
        "caller_name": call_state["caller_name"],
        "tools_used": call_state["tools_used"],
        "tools_blocked": call_state["tools_blocked"],
        "reminders": [dict(r) for r in db.reminders_for_call(call_state["call_id"])],
        "provider_down": call_state["provider_down"],
        "provider_limited": call_state.get("provider_limited", False),
        "stuck": call_state["stuck"],
    }


def run_call(caller_id="", caller_lines=None, owner_answers=None, verbose=True):
    """A whole call in one go: from the keyboard, or from a script (the evals).
    Just start_call, then handle_turn until the call ends, then finish_call."""
    opener = start_call(caller_id, caller_lines, owner_answers, verbose)
    hung_up = False
    if opener is not None:
        while True:
            user_text = next_caller_line()
            if user_text == "quit":
                hung_up = True
                break
            handle_turn(user_text)
            record_timing()  # agent stages only: no voice in the terminal or the evals
            if call_state["ended"]:
                break
    return finish_call(hung_up)


if __name__ == "__main__":
    caller_id = input("Caller ID (try 1111 for family, or press Enter): ").strip()
    run_call(caller_id=caller_id)