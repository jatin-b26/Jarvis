# Architecture

```
 caller speech (browser: hold the button or Space; the caller can talk over Jarvis at any time)
      |
      v
 speech-to-text      Sarvam saaras:v4, Roman-script output; ElevenLabs Scribe v3 as the fallback
      |
      v
 one turn            judge  (gpt-4.1-mini, runs alongside): "agreed to be put through", "reason known"
                     language: picked by code from the caller's words, so the voice never waits for the judge
                     agent  (gpt-5-mini, reasoning "low", streamed): 5 tools, picks what to do and say
      |
      v
 guards, in code     tool locks | leak guard | claim, promise, language and gender guards   every sentence, before it is spoken
      |
      v
 voice               ElevenLabs (Indian-accented voice "Sia"), sentence by sentence; fixed lines and fillers from a cache
      |
 saved notes  ---->  outbox (SQLite) -> background POST -> Tailscale -> n8n -> sheet -> WhatsApp
```

Every turn, tool call, guard decision, notification and timing is logged to SQLite and can be replayed
(`python show.py <call id>`).

## A full example call

A caller reaches Jarvis in the browser (hold the button or Space to talk). Each line below is marked with who
wrote it: **code** (a fixed line, same every call) or **model** (the wording varies).

```
Jarvis (code):   "Hello, Jarvis this side, Jatin ki AI assistant. Woh abhi phone nahi utha sakte.
                  Aap apna naam aur kis baare mein baat karni thi, bata sakte hain?"
Caller:          "Hi, this is Rahul. It's about the VPS renewal."
Jarvis (model):  "Thanks, Rahul. Should I try calling him, or take a message or set a reminder?"
                                           (name + reason known: the offer itself counts as the request)
Caller:          "Yes, try him."
  -> try_reach_owner         locks checked in code, then my phone rings: "Incoming call from Rahul: VPS renewal"
Jarvis (code):   "One sec, trying him now."                       (hold line while it rings)
                 ... I decline ...
Jarvis (code):   "Jatin couldn't take the call right now. Should I take a message or set a reminder for him?"
Caller:          "A reminder's fine, it's urgent."
  -> add_reminder            name and reason checked, saved; WhatsApp: "📞 Rahul called: VPS renewal, wants a callback (urgent)"
Jarvis (code):   "Okay, I have noted it for him. Anything else you want me to tell him?"
Caller:          "No, bye."
  -> end_call                outcome: reminder
Jarvis (code):   "Okay, thank you for calling. Take care, bye."
```

## Authority in code, judgment in the prompt

The model decides *what to do* and *how to say it*. Anything that must never go wrong is decided in Python,
where the model can't talk its way past it:

- **Ringing me:** the ring tool isn't on the model's menu until the caller has asked for me or Jarvis has
  offered to try me. It is refused unless the caller agreed to that offer or explicitly asked to be put through
  (read by the judge), a name and a reason are known (or the caller said there's no particular reason), and I
  haven't been rung yet on this call. The call connects only on my explicit "yes".
- **Saving:** junk names and reasons are rejected; a note that repeats one already saved is refused.
- **Call flow:** each tool at most once per turn; `end_call` can't ride along with the first save or with a ring.
- **Lines spoken by code, not the model:** the openers, the hold line, the "anything else?" follow-up, the
  decline and refusal lines, "connecting you now", short confirmations, every goodbye, the outage lines and
  the fillers. Most of these were once model lines that went wrong in a test; a fixed line can't.
- **Barge-in:** if the caller talks over Jarvis, audio stops, the rest of the reply is cancelled, and the
  history keeps only what the caller actually heard. A tool either completed before the interruption (kept,
  logged) or never started. Presses under 0.3 s are ignored, so an accidental tap doesn't cut Jarvis off.

The prompt handles tone, language (Roman Hinglish or English, following the caller), urgency, and which tool
fits what the caller said. The sentence-level honesty guards are described in [evals.md](evals.md).

## Family calls

Family members are recognised by caller ID and are never screened: my phone rings before Jarvis says a word.

- **I decline** (the Decline button): Jarvis opens with "Hello, Jarvis this side, Jatin ki AI assistant. Jatin
  abhi call nahi le paaye. Unhe callback ke liye remind kar doon, ya koi message dena hai?". There is no second
  ring after a decline, ever.
- **I don't pick up** (no answer within the ring time): "...Jatin ne abhi phone nahi uthaya. Ek baar aur try
  karoon, ya remind kar doon, ya koi message?". If they want another try, it rings **once** more.
- Any further ring request is refused with a fixed line. The locks are in code: at most 2 rings per family
  call, none after a decline, and connecting still needs my yes.
- Family reminders need no name or reason. Messages stay in the caller's own words in Roman script ("ghar
  jaldi aa jao"). A private mapping in `.env` (`FAMILY_NAMES=1111:Papa,...`) names them in notes, on my phone
  and on the page; it never enters the repo, and unmapped numbers read "Family member".

How the family flow is tested: [evals.md](evals.md#family-scenarios).

## WhatsApp notes

Every saved reminder or message goes to my WhatsApp, without slowing the call:

```
 Jarvis saves a note -> outbox row in jarvis.db (local write, ~10 ms) -> the call goes on
                        background worker -> Tailscale (encrypted) -> n8n webhook -> sheet -> WhatsApp
```

- **One JSON shape**, with a `note_id` (`jarvis-<call>-<type>-<n>`) made once when the note is saved, so a
  retry is recognisable and n8n can answer "duplicate" instead of sending it twice.
- **The text is the note as saved; the one-line summary is built by code, never rewritten by a model:**
  "📞 Rahul called: VPS renewal, wants a callback (urgent)", "🏠 Family call – Papa left a message: ghar
  jaldi aa jao".
- **Failures are kept, not lost:** anything but `200 {"saved": true}` (or `"duplicate": true`) stays in the
  outbox and is retried with backoff (10 s up to hourly, about two days), also after a server restart. Each
  attempt is logged (`notify_sent`, or `notify_failed` with 401 / 400 / 5xx / timeout / unreachable).
- **Off unless configured;** the shared secret travels only in a header and is never logged. Tests run against
  a stub webhook: success, a 500 then a retry with the same id, a duplicate reply, a 3 s webhook (the save
  took 9 ms), no URL set, and the family name mapping.

## Known issues and next steps

- **Watch items, counted in every eval run:** offers made in the same line as asking the reason (allowed:
  nothing rings without a reason); a family callback that also carries news saved as both a reminder and a
  message (allowed by the rule; 3 of 27 in the last family run); a lead-in in the middle of a family note
  ("Home is fine; caller asked you to call them").
- **The grader is noisy on refusals:** "I can't chat about the match" is flagged now and then (1 in 4
  re-scores of the same reply), although refusals are listed as not claims. Failures are read in the replay
  before any fix.
- **Third-person notes** still appear in 3-4 calls per full run ("Neha called; she asked me to tell you...").
- **Latency:** the next step is transcribing while the caller is still talking (deferred).
- **Demo scope:** browser push-to-talk, one call at a time, no telephony yet.
