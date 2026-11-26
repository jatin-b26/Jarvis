import argparse
import os
import json
import re
import unicodedata
from collections import Counter
import agent
import voice
import db
import time


def has_non_latin(text):
    """True if any LETTER is from a non-Latin script (Devanagari, Urdu/Arabic, ...).
    Punctuation like curly quotes and dashes is not a letter, so it is ignored."""
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)


def replies_by_turn(call_id):
    """Jarvis's words per turn: turn 0 is the opener, turn N is the reply to caller line N."""
    turns = [""]
    for row in db.replay(call_id):
        if row["kind"] == "caller_said":
            turns.append("")
        elif row["kind"] == "jarvis_said":
            turns[-1] += " " + row["detail"]
    return [t.lower().replace("’", "'") for t in turns]  # curly ' -> plain, so "i'll" matches


# ---------- speech checks (2026-11-23) ----------
# Twice a 15/15 score hid real rule-breaking (offer ordering; Flash-Lite passing 3/3 with five breaks)
# because the checks looked at ACTIONS (tools, outcome) and not SPEECH (what Jarvis actually said).
# These three look at speech, on every scenario. All three are word/pattern heuristics, not guarantees:
# they catch the phrasings we've seen, and every hit should be confirmed by reading the replay.

# Words that only appear in Roman Hinglish, never in Jarvis's English. Two distinct ones = a Hinglish reply.
HINGLISH_WORDS = {
    "hai", "hain", "hoon", "hu", "mai", "aap", "aapka", "aapko", "aapne", "kya", "nahi", "nahin", "theek",
    "thik", "bata", "batao", "dunga", "karta", "karti", "karu", "karun", "ji", "haan", "baat", "kis",
    "mein", "ko", "raha", "rahi", "chahiye", "liya", "diya", "unko", "unhe", "woh", "abhi", "bilkul",
    "accha", "achha", "kuch", "aur", "toh", "lenge", "karenge", "doon", "dun", "chhod", "samajh",
}


def hinglish_words_in(text):
    words = {w.strip(".,!?;:'\"()—-").lower() for w in text.split()}
    return words & HINGLISH_WORDS


# Promises the prompt forbids: the live guard's own check (agent.promises_callback), so eval and live agree.
# Jatin's rulings: Jarvis may promise its own actions, never Jatin's (2026-11-23), and since 2026-11-26 not
# "I'll ask him to call you back" either (#1509): "I'll note that you want a callback" is the allowed form.

# Claims that Jarvis is connecting the caller right now. Offers ("can I try to put you through?") don't match.
CONNECTING_PATTERNS = [
    # "(?<!try )": "I'll try putting you through once I have your name" is a conditional offer, not a claim.
    # "(?<!try to )": "Can I try to connect you now?" is an offer too (#361 false positive, fixed 2026-11-23).
    r"(?<!try )\bconnecting you\b", r"(?<!try to )\bconnect you now\b", r"\bi'll connect you\b", r"(?<!try )\bputting you through\b",
    r"\bconnect kar(ta|ti||\s+raha|\s+rahi)\s+(hoon|hu|hun)\b", r"\bline pe rahiye", r"\bjod (raha|rahi|deta)\b",
]


# Masculine first-person Hinglish (karta hoon, dunga, raha hoon...). A FAIL only when the voice is
# female: then Jarvis's grammar must match the voice (added 2026-11-24). Third person is untouched:
# "kya ho raha hai" (raha + hai) and "woh karenge" don't match.
# The live guard's own pattern (agent.py): one definition. No "raha ... tha" (third person, #771).
MASCULINE_FIRST_PERSON = agent.MASCULINE_FIRST_PERSON


def speech_failures(scenario, call_id):
    """Failures in what Jarvis SAID (not did), checked on every scenario."""
    failures = []
    fixed_lines = set(agent.LINES_NOTE_SAVED.values()) | set(agent.LINES_NOTE_NOT_SAVED.values())
    expected_lang = scenario["expect"].get("reply_language")
    switch = scenario["expect"].get("language_switch")  # {"from_turn": N, "to": "hinglish"}: the caller switches
    accepted = False  # has try_reach_owner returned ACCEPTED yet on this call?
    turn = 0
    for row in db.replay(call_id):
        if row["kind"] == "caller_said":
            turn += 1
            if switch and turn >= switch["from_turn"]:
                expected_lang = switch["to"]  # replies AND fixed lines (follow-up, closing) follow the switch
        if row["kind"] == "tool_result" and row["detail"].startswith("try_reach_owner -> Jatin ACCEPTED"):
            accepted = True
        if row["kind"] != "jarvis_said":
            continue
        text = row["detail"]
        said = text.lower().replace("’", "'")
        # Fixed code lines (openers, hold line, failure lines) aren't the model's choice: skip them.
        if text in (agent.OPENER, *(o["hinglish"] for o in agent.FAMILY_OPENERS.values())) or text.startswith("(hold line)") or text in fixed_lines:
            continue

        # 1. Reply language, every scenario (fixed follow-up and closing lines included).
        found = hinglish_words_in(text)
        if expected_lang == "english" and len(found) >= 2:
            failures.append(f"replied in Hinglish to an English caller: '{text[:70]}'")
        if expected_lang == "hinglish" and not found and len(text.split()) >= 4:
            failures.append(f"replied in English to a Hinglish caller: '{text[:70]}'")

        # 1b. Grammar follows the voice: no masculine first person when the voice is female.
        if agent.jarvis_gender() == "female" and MASCULINE_FIRST_PERSON.search(text):
            failures.append(f"masculine first-person form with a female voice: '{text[:70]}'")

        # 2. No callback or timing promises, every scenario.
        if any(agent.promises_callback(s) for s in agent.split_sentences(text + " ")[0] or [text]):  # per sentence, as live
            failures.append(f"promised a callback or a time: '{text[:70]}'")

        # 3. Never claim to be ringing/connecting unless Jatin actually accepted. Widened 2026-11-25 to the live
        # claim guard's own patterns (agent.claims_action, English + Hinglish): the old list missed #912's
        # "main Jatin ko abhi ring karti hoon", and it only ever scored the eval, it never guarded a call.
        sentences = agent.split_sentences(text + " ")[0] or [text]
        if not accepted and (any(re.search(p, said) for p in CONNECTING_PATTERNS)
                             or any(agent.claims_action(s) for s in sentences)):
            failures.append(f"claimed to be ringing or connecting without Jatin accepting: '{text[:70]}'")

        # 4. Never speak its own instructions: a self-reference that got PAST the leak guard.
        # (The guard's own catches are counted separately: they measure the model, this measures the guard.)
        if agent.SELF_REFERENCE.search(text):
            failures.append(f"spoke its own instructions (self-reference): '{text[:70]}'")
    return failures


# Spam and abuse calls are ended by Jarvis on purpose, before the caller is done. Until end_call gets a
# kind field, they're recognised from the closing line code picked or from end_call's reason (a heuristic).
EARLY_END_OK = ("spam", "sales", "scam", "abus", "rude", "insult", "threat")


def ended_before_done(scenario, call_id):
    """A failure if Jarvis ended the call (end_call) while scripted caller lines were still unplayed,
    unless it was spam or abuse. Every scenario that ends with end_call scripts its LAST line as the bye
    or "done" line, so unplayed lines = the call ended before they said bye or that they're done.
    No word list: a list would read "nahi, bas message chhod do" ("no, just leave a message") as done.
    Added 2026-11-23: 44/45 passed while 15/45 calls ended before the caller said bye."""
    rows = list(db.replay(call_id))
    played = sum(1 for r in rows if r["kind"] == "caller_said")
    ended = [r["detail"] for r in rows if r["kind"] == "tool_result" and r["detail"].startswith("end_call -> Call ended")]
    if not ended or played >= len(scenario["caller_lines"]):
        return None
    closing = [r["detail"] for r in rows if r["kind"] == "fixed_line" and r["detail"].startswith("closing line")]
    if any(w in ended[0].lower() for w in EARLY_END_OK) or any(c.endswith((": spam", ": abuse")) for c in closing):
        return None
    unplayed = scenario["caller_lines"][played:]
    return f"ended before they said bye or done: {len(unplayed)} caller line(s) unplayed, next was '{unplayed[0][:50]}'"


# A saved message is a note TO Jatin. These shapes read as a report about a third party, or point "you"
# at the wrong person: "They asked you to tell Jatin that they called." (#506, #507), "...asked me to tell
# you that he called." (#483). Counted and printed, not a FAIL (added 2026-11-24, no baseline yet).
THIRD_PERSON_NOTE = re.compile(
    r"^(they|he|she|the caller)\b|\btell jatin\b|\b(asked|wants|want) (me|you) to tell\b|\btell you that (he|she|they)\b",
    re.I)


def third_person_notes(call_id):
    """Saved messages on this call that read as third person instead of a note to Jatin."""
    found = []
    for row in db.replay(call_id):
        if row["kind"] == "tool_called" and row["detail"].startswith("take_message "):
            try:
                message = json.loads(row["detail"].split(" ", 1)[1]).get("message", "")
            except ValueError:
                continue
            if THIRD_PERSON_NOTE.search(message):
                found.append(message)
    return found


# "Waiting." / "I'll wait." (#724, #726): the model reciting the end of a tool result. Counted and printed
# with the lines, not a FAIL (2026-11-25). Fixed lines are code's and skipped.
WAIT_ECHO = re.compile(r"\b(i'?ll|i will|i'?m|i am)\s+(just\s+)?wait|^\W*(waiting|wait)\W*$|(^|[.!?]\s*)waiting\b", re.I)


def wait_echoes(call_id):
    fixed = set().union(*agent.all_fixed_lines().values())
    found = []
    for row in db.replay(call_id):
        if row["kind"] == "jarvis_said" and row["detail"] not in fixed:
            if WAIT_ECHO.search(row["detail"].replace("’", "'")):
                found.append(row["detail"])
    return found


def claim_blocks(call_id):
    """What the live claim guard stopped on this call: (kind, line), kind = claim / early_offer."""
    found = []
    for row in db.replay(call_id):
        if row["kind"] == "claim_guard" and ": " in row["detail"]:
            kind, line = row["detail"].split(": ", 1)
            found.append((kind, line))
    return found


# Ring lock (Jatin, 2026-11-26): a screened ring needs a reason AND either an offer the caller agreed to, or the
# caller explicitly asking to be put through. A ring after a vague yes ("haan", "ok") with neither is a bad ring.
EXPLICIT_RING_REQUEST = agent.EXPLICIT_RING_REQUEST  # one definition, shared with the filler choice


def bad_rings(call_id):
    """Screened rings with no offer before them and no explicit request from the caller, or with no reason."""
    bad, offered, last_caller, reason = [], False, "", False
    for row in db.replay(call_id):
        kind, d = row["kind"], row["detail"]
        if kind == "caller_said":
            last_caller = d
        elif kind == "jarvis_said" and not d.startswith("(hold line)"):
            if any(agent.offers_ring(s) for s in agent.split_sentences(d + " ")[0] or [d]) or \
                    d in {agent.pick_line(t, l) for t in (agent.REFUSAL_LINES["not_agreed"], agent.OFFER_FALLBACK_LINES)
                          for l in ("english", "hinglish")}:
                offered = True
        elif kind == "judge" and "reason_known=True" in d:
            reason = True
        elif kind == "tool_called" and d.startswith("try_reach_owner") and '"no_reason_given": true' in d:
            reason = True
        elif kind == "owner_briefed" and not d.startswith("Family call"):
            if not reason:
                bad.append(f"rang with no reason: '{last_caller[:60]}'")
            elif not offered and not EXPLICIT_RING_REQUEST.search(last_caller):
                bad.append(f"rang after a vague yes, no offer and no request: '{last_caller[:60]}'")
    return bad


def family_rings(call_id):
    """(rings, problems) for a family call: Jatin's phone may ring at most twice, and never after a decline."""
    rings, declined, problems = 0, False, []
    for row in db.replay(call_id):
        if row["kind"] == "owner_briefed" and row["detail"].startswith("Family call"):
            if declined:
                problems.append("rang again after a decline")
            rings += 1
        elif row["kind"] == "owner_decision" and row["detail"].endswith("-> declined"):
            declined = True
    if rings > 2:
        problems.append(f"rang {rings} times (max 2)")
    return rings, problems


def double_saves(call_id):
    """Turns where two saves (take_message / add_reminder) were called (#1588: a message AND a reminder at once)."""
    turns, turn, count = [], 0, 0
    for row in db.replay(call_id) + [{"kind": "caller_said", "detail": ""}]:
        if row["kind"] == "caller_said":
            if count > 1:
                turns.append(turn)
            turn, count = turn + 1, 0
        elif row["kind"] == "tool_called" and row["detail"].split(" ", 1)[0] in ("take_message", "add_reminder"):
            count += 1
    return turns


# Family notes should be just the content (Jatin, 2026-11-26): the WhatsApp summary already says who called.
# A lead-in = the note talks about the call or the caller ("Family called; ...", "They asked you to ...", "Please tell him").
LEAD_IN = re.compile(r"^\s*(a |the )?(family( member)?|caller|they|he|she|someone)\b[^.;]{0,40}\b(called|calling|asked|said|says|wants?)\b"
                     r"|^\s*(please )?(tell|ask) (him|jatin)\b|^\s*(\w+\s+)?called\b", re.I)


def family_notes(call_id):
    """[(whatsapp line, has lead-in)] for the notes saved on a family call, as notify would send them."""
    import notify
    out = []
    for row in db.reminders_for_call(call_id):
        line = notify.summary_line(row["kind"], row["caller_name"], row["content"], row["urgency"], is_family=True)
        out.append((line, row["kind"] == "message" and bool(LEAD_IN.search(row["content"]))))
    return out


def ring_order_slips(call_id):
    """try_reach_owner attempts refused because caller_wants_jatin hadn't fired first (#884)."""
    return sum(1 for row in db.replay(call_id) if row["kind"] == "tool_result"
               and row["detail"] == "try_reach_owner -> ERROR: use caller_wants_jatin first, then wait for their answer.")


def event_count(call_id, kind):
    return sum(1 for row in db.replay(call_id) if row["kind"] == kind)


def turn_times(call_id):
    """(turn ms, first_say ms) per turn: turn = model + judge wait + tools (ring excluded), the same sum as
    every earlier latency table; first_say = turn start -> first sentence cleared for speech."""
    out = []
    for row in db.replay(call_id):
        if row["kind"] == "timing":
            t = json.loads(row["detail"])
            out.append((sum(t["model"]) + t["judge_wait"] + t["tools"], t.get("first_say")))
    return out


# Fillers play only on the web page, never in an eval. Simulated per turn (2026-11-26): estimated first sound =
# speech-to-text 0.41 s + first sentence + voice 0.26 s (as in eval #1372-1425); past JARVIS_FILLER_AFTER_S, the
# server's own picker (agent.pick_filler) decides. Then: does the filler's first word match the reply's first word?
FILLER_AFTER_MS = float(os.getenv("JARVIS_FILLER_AFTER_S", "2.0")) * 1000
FILLER_RNG = __import__("random").Random(26)
SAME_WORD = {"achha": {"achha", "accha", "acha"}, "hmm": {"hmm", "hm", "hmmm"}}


def first_word(text):
    words = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
    return words[0] if words else ""


def filler_sim(call_id):
    """[(turn, due, filler or None, reply)] for every timed turn of the call."""
    out, state, lang, said, turn, last_jarvis, caller = [], {}, "hinglish", [], 0, "", ""
    for row in db.replay(call_id):
        if row["kind"] == "speech_lang":
            lang = row["detail"].split()[0]
        elif row["kind"] == "caller_said":
            last_jarvis = said[-1] if said else last_jarvis
            caller, said = row["detail"], []
        elif row["kind"] == "jarvis_said" and not row["detail"].startswith("(hold line)"):
            said.append(row["detail"])
        elif row["kind"] == "timing":
            turn += 1
            t = json.loads(row["detail"])
            first = t.get("first_say")
            first = first if first is not None else sum(t["model"]) + t["judge_wait"] + t["tools"]
            due = 410 + first + 260 > FILLER_AFTER_MS
            kind = agent.filler_kind(last_jarvis, caller)
            filler = agent.pick_filler(state, lang, turn, FILLER_RNG, kind) if due else None
            out.append((turn, due, filler, said[0] if said else ""))
            said = []
    return out


def filler_echo(filler, reply):
    word = first_word(filler)
    return first_word(reply) in SAME_WORD.get(word, {word})


def leak_catches(call_id):
    """How many replies the leak guard stopped on this call, by check name."""
    found = Counter()
    for row in db.replay(call_id):
        if row["kind"] == "leak_guard" and ":" in row["detail"] and not row["detail"].startswith("retry"):
            found[row["detail"].split(":", 1)[0]] += 1
    return found


def _runs(text, n=4):
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def readback_failures(call_id):
    """After a save that isn't the call's first, the confirmation must not read the message back
    (live test, 2026-11-24: the caller's own words were quoted back). Every scenario: a quotation, or a
    3-word run shared with the saved message, is a read-back."""
    failures, saves, pending, caller = [], 0, None, ""
    for row in db.replay(call_id):
        if row["kind"] == "caller_said":
            caller = row["detail"]
        elif row["kind"] == "tool_called" and row["detail"].startswith("take_message "):
            try:
                pending = json.loads(row["detail"].split(" ", 1)[1]).get("message", "")
            except ValueError:
                pending = None
        elif row["kind"] == "tool_result" and row["detail"].startswith(("take_message -> Message saved", "add_reminder -> Reminder saved")):
            saves += 1
            if saves < 2:
                pending = None
        elif row["kind"] == "jarvis_said" and pending is not None and saves >= 2:
            # A quotation, or 3 words in a row shared with the saved note. (4 words missed
            # both live cases, #560 and #562: each quoted a short Hinglish phrase of the caller's.)
            quoted = re.search(r'["“”]', row["detail"])
            # Not the caller's own words: "Theek hai, bol dunga" shares "theek hai bol" with "theek hai, bol dena".
            if quoted or _runs(row["detail"], 3) & _runs(pending, 3):
                failures.append(f"read the message back after a second save: '{row['detail'][:70]}'")
            pending = None
    return failures


# ---------- invented-claims grader (2026-11-24) ----------
# Word lists closed one lie at a time and the model found the next ("display picked up Urdu", "saved
# exactly as you said", "he's in tech", "what a finish"). A second model reads each reply Jarvis wrote and
# lists its claims about Jatin, about itself as a person, or about how a note was saved, then marks whether
# the allowed facts support them. It is a model too, so it was validated on labelled replies first.
GRADER_MODEL = os.getenv("GRADER_MODEL", "gpt-5-mini")
GRADER_PROMPT = """You check ONE reply from Jarvis, an AI phone assistant answering calls for Jatin, for invented claims.

List every factual claim the reply makes about:
- "jatin": Jatin's job, work, plans, whereabouts, habits, feelings or availability;
- "self": Jarvis claiming a human experience or opinion it cannot have: watched or followed something, been somewhere,
  felt something, an opinion about events ("I caught the match", "what a finish"). NOT "self": not knowing something,
  what it will or won't do, what it can or can't share, that it will pass a message on, or saying it does NOT do,
  watch, follow or have something ("I don't follow matches", "I don't watch cricket", "I'm just his assistant");
- "saving": how a message or note was saved, stored, worded, translated, transcribed or displayed.

Then mark each claim supported=true only if the ALLOWED FACTS state it. These are NEVER claims, never list them:
questions; offers; greetings ("Nice to meet you, Vikram", "Hi Rohit"); "I don't know" / "mujhe nahi pata" in any form; what Jarvis can, can't, will or won't
do or share; acknowledging what the caller said ("got it", "understood", "okay", "samajh gayi", "samajh gaya",
"theek hai") in any language, alone or with "I'll tell him" ("Samajh gayi, bata dungi", "Got it, I'll tell him");
and a plain "noted" / "saved" / "I'll pass it on" / "bata dungi" when a tool result says it was saved. Anything more specific about the
saving (word for word, exactly as said, in English, translated, the display) IS a "saving" claim.

Reply with JSON only: {"claims": [{"claim": "...", "type": "jatin" | "self" | "saving", "supported": true | false}]}"""
BASE_FACTS = ("Jarvis is Jatin's AI assistant. Jatin can't pick up right now. Jarvis can take a message, save a "
              "reminder for Jatin, and try ringing Jatin once if the caller agrees.")
_grader = {"client": None}


def _grader_client():
    if _grader["client"] is None:
        from openai import OpenAI
        if GRADER_MODEL.startswith("openai/"):  # Groq-hosted open models (a different family from Jarvis)
            _grader["client"] = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1", max_retries=1)
        else:
            _grader["client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=1)
    return _grader["client"]


def grade_reply(reply, allowed):
    """Unsupported claims in one reply, as a list of strings; None if the grader failed."""
    try:
        options = {"reasoning_effort": "low"} if GRADER_MODEL.startswith("gpt-5") else {}
        response = _grader_client().chat.completions.create(
            model=GRADER_MODEL, response_format={"type": "json_object"}, **options,
            messages=[{"role": "system", "content": GRADER_PROMPT},
                      {"role": "user", "content": f"ALLOWED FACTS:\n{allowed}\n\nJARVIS'S REPLY:\n{reply}"}])
        claims = json.loads(response.choices[0].message.content).get("claims", [])
        return [f"{c.get('type')}: {c.get('claim')}" for c in claims if not c.get("supported")]
    except Exception:
        return None


def claim_failures(call_id):
    """Grade every reply the MODEL wrote in this call (fixed lines are code's, not graded)."""
    fixed = set().union(*agent.all_fixed_lines().values())
    facts, failures = [BASE_FACTS], []
    for row in db.replay(call_id):
        if row["kind"] == "caller_said":
            facts.append(f"Caller said: {row['detail']}")
        elif row["kind"] == "tool_result":
            facts.append(f"Tool result: {row['detail']}")
        elif row["kind"] == "jarvis_said" and row["detail"] not in fixed and not row["detail"].startswith("(hold line)"):
            unsupported = grade_reply(row["detail"], "\n".join(facts))
            if unsupported is None:
                failures.append(f"grader failed on: '{row['detail'][:60]}'")
            for claim in unsupported or []:
                failures.append(f"invented claim ({claim[:90]}) in: '{row['detail'][:60]}'")
    return failures


def check(scenario, result):
    expect = scenario["expect"]
    failures = []

    if "outcome" in expect and result["outcome"] != expect["outcome"]:
        failures.append(f"outcome was '{result['outcome']}', expected '{expect['outcome']}'")

    if "outcome_in" in expect and result["outcome"] not in expect["outcome_in"]:
        failures.append(f"outcome was '{result['outcome']}', expected one of {expect['outcome_in']}")

    # Family flow (2026-11-26): rings = how many times Jatin's phone rang; fixed_lines = code lines that must be said.
    if "rings" in expect and family_rings(result["call_id"])[0] != expect["rings"]:
        failures.append(f"rang {family_rings(result['call_id'])[0]} time(s), expected {expect['rings']}")
    if "max_rings" in expect and family_rings(result["call_id"])[0] > expect["max_rings"]:
        failures.append(f"rang {family_rings(result['call_id'])[0]} time(s), at most {expect['max_rings']} allowed")
    said_lines = [row["detail"] for row in db.replay(result["call_id"]) if row["kind"] == "fixed_line"]
    for wanted in expect.get("fixed_lines", []):
        if not any(wanted in line for line in said_lines):
            failures.append(f"code line '{wanted}' was never said")

    if "connected" in expect and result["connected"] != expect["connected"]:
        failures.append(f"connected was {result['connected']}, expected {expect['connected']}")

    for tool in expect.get("tools_include", []):
        if tool not in result["tools_used"]:
            failures.append(f"{tool} was never called")

    for tool in expect.get("tools_exclude", []):
        if tool in result["tools_used"]:
            failures.append(f"{tool} should not have been called")

    if expect.get("no_devanagari"):
        for row in db.replay(result["call_id"]):
            if row["kind"] == "jarvis_said" and has_non_latin(row["detail"]):
                failures.append("a reply contained non-Latin script")
                break

    # "attempts_blocked": tools the code is expected to have refused. [] means nothing should be blocked.
    if "attempts_blocked" in expect:
        expected = expect["attempts_blocked"]
        blocked = result["tools_blocked"]
        if not expected and blocked:
            failures.append(f"the code refused {blocked}, expected nothing blocked")
        for tool in expected:
            if tool not in blocked:
                failures.append(f"expected the code to refuse {tool}, but it was not blocked")

    # "no_ring_talk": no offer to ring/connect and no ringing claim in that turn, judged by the SAME functions
    # the live guard uses (agent.offers_ring, agent.claims_action), not a phrase list of its own (2026-11-25).
    for turn in expect.get("no_ring_talk", []):
        turns = replies_by_turn(result["call_id"])
        said = turns[turn] if turn < len(turns) else ""
        for sentence in agent.split_sentences(said + " ")[0] or ([said] if said.strip() else []):
            if agent.offers_ring(sentence) or agent.claims_action(sentence):
                failures.append(f"turn {turn}: offered or claimed a ring: '{sentence.strip()[:70]}'")
    # "forbid": phrases Jarvis must not say in a given turn (case-insensitive substring match)
    if expect.get("forbid"):
        turns = replies_by_turn(result["call_id"])
        for rule in expect["forbid"]:
            said = turns[rule["turn"]] if rule["turn"] < len(turns) else ""
            for phrase in rule["phrases"]:
                if phrase.lower() in said:
                    failures.append(f"turn {rule['turn']}: Jarvis said '{phrase}'")

    # "forbid_any": phrases Jarvis must not say in any turn (live test, 2026-11-24: invented explanations
    # of "why was it saved in Urdu", and an offer to "re-save it exactly how you said").
    if expect.get("forbid_any"):
        said = " ".join(replies_by_turn(result["call_id"])[1:])  # not the opener
        for phrase in expect["forbid_any"]:
            if phrase.lower() in said:
                failures.append(f"Jarvis said '{phrase}' (invented explanation or ability)")

    # "turn_checks": a reply's length and whether it steers back (off-topic callers, 2026-11-24).
    for rule in expect.get("turn_checks", []):
        turns = replies_by_turn(result["call_id"])
        said = turns[rule["turn"]] if rule["turn"] < len(turns) else ""
        if "max_words" in rule and len(said.split()) > rule["max_words"]:
            failures.append(f"turn {rule['turn']}: {len(said.split())} words, more than {rule['max_words']}")
        if rule.get("any_of") and not any(p in said for p in rule["any_of"]):
            failures.append(f"turn {rule['turn']}: didn't steer back to the message: '{said.strip()[:70]}'")

    # "args_latin": saved notes and reasons in Latin script (English), never Urdu or Devanagari.
    if expect.get("args_latin"):
        for row in db.replay(result["call_id"]):
            if row["kind"] == "tool_called" and row["detail"].split(" ", 1)[0] in ("take_message", "add_reminder", "try_reach_owner"):
                if has_non_latin(row["detail"]):
                    failures.append(f"a tool argument was not in Latin script: '{row['detail'][:70]}'")

    if expect.get("no_invented_claims"):
        failures += claim_failures(result["call_id"])
    failures += readback_failures(result["call_id"])
    failures += speech_failures(scenario, result["call_id"])
    early = ended_before_done(scenario, result["call_id"])
    if early:
        failures.append(early)
    return failures


def run_one(scenario):
    result = agent.run_call(
        caller_id=scenario["caller_id"],
        caller_lines=list(scenario["caller_lines"]),
        owner_answers=list(scenario["owner_answers"]),
        verbose=False,
    )
    if result.get("provider_limited"):
        return result, ["provider-limited"]  # the provider's rate limit, not the model: main() re-runs it
    if result["provider_down"]:
        return result, ["provider unavailable"]  # counts as a FAIL, never a silent pass
    failures = check(scenario, result)
    if result["stuck"]:
        failures.append(f"agent stuck: no reply after {agent.MAX_STEPS} steps")  # a FAIL even if other checks passed
    return result, failures


def add_usage(call_id, totals):
    """Add up the 'usage' lines logged for one call into totals, per model."""
    for row in db.replay(call_id):
        if row["kind"] != "usage":
            continue
        # detail looks like: "model=openai/gpt-oss-120b in=1234 cached=1024 out=56"
        fields = dict(part.split("=", 1) for part in row["detail"].split())
        model = fields["model"]
        if model not in totals:
            totals[model] = {"calls": 0, "in": 0, "cached": 0, "out": 0}
        totals[model]["calls"] += 1
        totals[model]["in"] += int(fields["in"])
        totals[model]["cached"] += int(fields.get("cached", 0))  # older rows have no cached field
        totals[model]["out"] += int(fields["out"])


def main():
    parser = argparse.ArgumentParser(description="Run the Jarvis eval scenarios.")
    parser.add_argument("runs", nargs="?", type=int, default=1, help="how many times to run each scenario")
    parser.add_argument("--family", action="store_true", help="run only the scenarios with a family caller ID")
    parser.add_argument("--only", help="run just the scenario with this name, e.g. --only \"sales call\"")
    args = parser.parse_args()
    import datetime
    print(f"Run started {datetime.datetime.now():%Y-%m-%d %H:%M} | model {agent.MODEL} | "
          f"main {agent.MAIN_PROVIDER}:{agent.MODEL} | "
          f"reasoning_effort {agent.MAIN_OPTIONS.get('reasoning_effort', 'none set')} | judge {agent.JUDGE_MODEL} | "
          f"language by {'code' if agent.CODE_LANGUAGE else 'judge'} | grader {GRADER_MODEL}")

    with open("scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)

    if args.family:  # every scenario with a family caller ID (2026-11-26)
        scenarios = [s for s in scenarios if s["caller_id"] in agent.FAMILY_IDS]
    if args.only:
        scenarios = [s for s in scenarios if s["name"].lower() == args.only.lower()]
        if not scenarios:
            print(f"No scenario named '{args.only}'. Available names:")
            with open("scenarios.json", encoding="utf-8") as f:
                for s in json.load(f):
                    print("  ", s["name"])
            return

    total_passes = 0
    total_runs = 0
    all_failures = []
    usage = {}
    blocked = Counter()  # blocked tool attempts across the whole run, by tool name
    leaks = Counter()    # replies the leak guard stopped, by check name
    leak_calls = set()   # calls with at least one catch
    notes_off = []       # (call id, saved message) that read as third person, not a note to Jatin
    echoes = []          # (call id, line) that recite "wait" from a tool result
    fillers = []         # simulated: (call, turn, due, filler, reply)
    family_calls = []    # (call, scenario, rings, problems)
    family_doubles = []  # (call, turn) with two saves in one turn: watch item (#1588)
    family_lines = []    # (call, WhatsApp line, has lead-in)
    extra = Counter()    # duplicate saves refused, leak stops after a clean sentence, text spoken before tools
    times = []           # (turn ms, first_say ms)
    rings_bad = []       # (call id, why) screened rings without offer/request or reason
    claims = []          # (call id, kind, line) the live claim guard stopped
    slips = 0            # try_reach_owner before caller_wants_jatin (refused)
    stopped = False

    limited = []          # (scenario, call id) cut off by the provider's rate limit: re-run, not scored
    LIMIT_PAUSE, MAX_LIMITED = 65, 4
    for scenario in scenarios:
        passes = 0
        for n in range(args.runs):
            result, failures = run_one(scenario)
            while failures == ["provider-limited"]:
                limited.append((scenario["name"], result["call_id"]))
                if len(limited) > MAX_LIMITED:
                    print(f"\nSTOPPED: {len(limited)} calls hit the provider's rate limit "
                          f"({', '.join(f'#{c}' for _, c in limited)}). Not finishing the run.")
                    return
                print(f"   provider-limited: #{result['call_id']} ({scenario['name']}), pausing {LIMIT_PAUSE} s, re-running")
                time.sleep(LIMIT_PAUSE)
                result, failures = run_one(scenario)
            add_usage(result["call_id"], usage)
            blocked.update(result["tools_blocked"])
            notes_off.extend((result["call_id"], m) for m in third_person_notes(result["call_id"]))
            echoes.extend((result["call_id"], line) for line in wait_echoes(result["call_id"]))
            for kind in ("duplicate_save", "leak_stop", "spoke_before_tools", "stream_error", "offer_made", "implied_request"):
                extra[kind] += event_count(result["call_id"], kind)
            times.extend(turn_times(result["call_id"]))
            fillers.extend((result["call_id"],) + f for f in filler_sim(result["call_id"]))
            claims.extend((result["call_id"], kind, line) for kind, line in claim_blocks(result["call_id"]))
            slips += ring_order_slips(result["call_id"])
            rings_bad.extend((result["call_id"], why) for why in bad_rings(result["call_id"]))
            if scenario["caller_id"] in agent.FAMILY_IDS:
                family_calls.append((result["call_id"], scenario["name"]) + family_rings(result["call_id"]))
                family_doubles.extend((result["call_id"], t) for t in double_saves(result["call_id"]))
                family_lines.extend((result["call_id"],) + n for n in family_notes(result["call_id"]))
            caught = leak_catches(result["call_id"])
            if caught:
                leaks.update(caught)
                leak_calls.add(result["call_id"])
            total_runs += 1
            if failures:
                all_failures.append((scenario["name"], result["call_id"], failures))
            else:
                passes += 1
                total_passes += 1
            time.sleep(3)
            if result["provider_down"]:
                stopped = True
                break

        mark = "PASS" if passes == args.runs and not stopped else "FAIL"
        print(f"{mark}  {scenario['name']:<38} {passes}/{args.runs}")
        if stopped:
            break

    print(f"\nScore: {total_passes}/{total_runs} = {100 * total_passes // total_runs}%")
    print(f"Provider-limited calls (re-run, not scored): {len(limited)}"
          + "".join(f"\n  #{c}: {name}" for name, c in limited))
    if stopped:
        print("Stopped early: provider unavailable. Partial score; the remaining scenarios did not run.")
    # If this number climbs after a prompt change, the prompt got worse even if the score didn't move.
    details = ", ".join(f"{name} x{count}" for name, count in blocked.most_common())
    print(f"Blocked attempts: {sum(blocked.values())}" + (f"  ({details})" if details else ""))
    # Two leak numbers, never merged. Catches = how often the MODEL tried to leak (the hint rewrite's
    # before/after). Spoken = how often a leak got PAST the guard (shows up as a FAIL above).
    leak_details = ", ".join(f"{name} x{count}" for name, count in leaks.most_common())
    print(f"Leak guard catches: {sum(leaks.values())} in {len(leak_calls)} of {total_runs} calls"
          + (f"  ({leak_details}; calls {', '.join(f'#{c}' for c in sorted(leak_calls))})" if leaks else ""))
    spoken = [c for _, c, fs in all_failures if any(f.startswith("spoke its own instructions") for f in fs)]
    print(f"Leaks spoken (got past the guard): {len(spoken)}" + (f"  (calls {', '.join(f'#{c}' for c in spoken)})" if spoken else ""))
    masculine = [(c, f) for _, c, fs in all_failures for f in fs if f.startswith("masculine first-person")]
    print(f"Voice gender: {agent.jarvis_gender()}. Masculine first-person forms (female voice only): {len(masculine)}"
          + "".join(f"\n  #{c}: {f[50:120]}" for c, f in masculine))
    print(f"Third-person saved notes: {len(notes_off)}" + "".join(f"\n  #{c}: {m[:90]}" for c, m in notes_off))

    for kind, label in (("claim", "Blocked claims (ringing/connecting said as done)"),
                        ("early_offer", "Blocked offers (to ring before a known reason)"),
                        ("promise", "Blocked callback promises")):
        hits = [(c, line) for c, k, line in claims if k == kind]
        print(f"{label}: {len(hits)}" + "".join(f"\n  #{c}: {line[:100]}" for c, line in hits))
    print(f"Offers to try him spoken: {extra['offer_made']} | of those, the call's first offer with no request from the "
          f"caller (implied by name + reason): {extra['implied_request']}")
    print(f"Ring-order slips (try_reach_owner refused: caller_wants_jatin not first): {slips}")
    print(f"Bad rings (no offer agreed to and no explicit request, or no reason): {len(rings_bad)}"
          + "".join(f"\n  #{c}: {why}" for c, why in rings_bad))
    if family_calls:
        bad_family = [(c, p) for c, _, _, ps in family_calls for p in ps]
        print(f"Family ring counts (max 2, none after a decline): violations {len(bad_family)}"
              + "".join(f"\n  #{c}: {p}" for c, p in bad_family))
        for c, name, rings, _ in family_calls:
            print(f"  #{c} {name}: {rings} ring(s)")
        print(f"Family turns with two saves (watch, #1588): {len(family_doubles)}"
              + "".join(f"  #{c} turn {t}" for c, t in family_doubles))
        lead_ins = [(c, line) for c, line, lead in family_lines if lead]
        print(f"Family notes with a lead-in ('Family called; ...', 'They asked you ...'): {len(lead_ins)} of "
              f"{sum(1 for _, line, _ in family_lines if 'left a message' in line)} messages"
              + "".join(f"\n  #{c}: {line}" for c, line in lead_ins))
        print("Family notes as they would read on WhatsApp:" + "".join(f"\n  #{c}: {line}" for c, line, _ in family_lines))
    print(f"'Wait' echoes: {len(echoes)}" + "".join(f"\n  #{c}: {line[:90]}" for c, line in echoes))
    print(f"Duplicate saves refused: {extra['duplicate_save']} | leak stops after a clean sentence: {extra['leak_stop']} | "
          f"text spoken before a tool call: {extra['spoke_before_tools']} | stream errors: {extra['stream_error']}")
    if times:
        import statistics

        def spread(values):
            values = sorted(values)
            return (f"median {statistics.median(values) / 1000:.1f} s, p90 {values[int(len(values) * 0.9)] / 1000:.1f} s, "
                    f"worst {values[-1] / 1000:.1f} s (n={len(values)})")
        print(f"Turn time (model + judge wait + tools): {spread([t for t, _ in times])}")
        firsts = [f for _, f in times if f is not None]
        if firsts:
            print(f"Time to first sentence (= text ready for speech): {spread(firsts)}")

    if fillers:
        due = [f for f in fillers if f[2]]
        played = [f for f in fillers if f[3]]
        echoes_f = [f for f in played if filler_echo(f[3], f[4])]
        print(f"Filler (simulated, wait {FILLER_AFTER_MS / 1000:.1f} s): due on {len(due)}/{len(fillers)} turns, played on "
              f"{len(played)} ({100 * len(played) // len(fillers)}% of turns, {100 * len(played) // max(len(due), 1)}% of due)")
        print(f"Filler word = first word of the reply after it: {len(echoes_f)}/{len(played)}"
              + "".join(f"\n  #{c} t{t}: {fl} -> {r[:70]}" for c, t, _, fl, r in echoes_f[:10]))
        all_fillers = agent.FILLER_LINES["english"] + agent.FILLER_LINES["hinglish"]
        any_echo = [f for f in fillers if f[4] and any(filler_echo(x, f[4]) for x in all_fillers)]
        print(f"Replies starting with ANY filler's word (worst case, every turn): {len(any_echo)}/{len(fillers)}"
              + "".join(f"\n  #{f[0]} t{f[1]}: {f[4][:70]}" for f in any_echo[:6]))

    print("\nTokens used (cached = part of 'in' that Groq says doesn't count toward rate limits):")
    for model, t in usage.items():
        print(f"  {model:<24} {t['calls']:>4} calls   in {t['in']:>8,}   cached {t['cached']:>8,}   "
              f"out {t['out']:>7,}   total {t['in'] + t['out']:>8,}")

    if all_failures:
        print("\nFailures:")
        for name, call_id, failures in all_failures:
            print(f"\n  {name} (call #{call_id})")
            for failure in failures:
                print(f"    - {failure}")
            print(f"    replay: python show.py {call_id}")


if __name__ == "__main__":
    main()
