# Jarvis: an inbound call-screening voice agent

When I don't pick up, Jarvis answers in Hinglish or English, finds out who's calling and why, and takes a
message, saves a reminder, or rings me and connects only if I say yes. Every saved note reaches me on WhatsApp.

## Demo

Each line is marked with who wrote it: **code** (a fixed line, same every call) or **model** (the wording varies).

```
Jarvis (code):   "Hello, Jarvis this side, Jatin ki AI assistant. Woh abhi phone nahi utha sakte.
                  Aap apna naam aur kis baare mein baat karni thi, bata sakte hain?"
Caller:          "Hi, this is Rahul. It's about the VPS renewal."
Jarvis (model):  "Thanks, Rahul. Should I try calling him, or take a message or set a reminder?"
Caller:          "Yes, try him."            -> my phone rings with a briefing ... I decline
Jarvis (code):   "Jatin couldn't take the call right now. Should I take a message or set a reminder for him?"
Caller:          "A reminder's fine, it's urgent."
  -> add_reminder   WhatsApp: "📞 Rahul called: VPS renewal, wants a callback (urgent)"
Jarvis (code):   "Okay, I have noted it for him. Anything else you want me to tell him?"
```

## Headline results

- **Latency: 15.9 s → 4.3 s** median per turn, like for like (same scenarios, same measure).
- **Eval: 74/75** in the latest full run (25 scenarios x3, each call checked on actions *and* speech; 71/75 as
  run, re-scored after fixing one of my own scenario scripts).
- **Speech-to-text bake-off on my own voice:** ~24 → 2 words wrong out of 98.
- **Rejected faster models** that answered in ~1.1 s but broke rules.

## Architecture

```
 caller speech (browser, hold to talk; the caller can talk over Jarvis at any time)
   -> speech-to-text   Sarvam saaras:v4 (Roman script); ElevenLabs Scribe v3 as fallback
   -> one turn         agent gpt-5-mini (5 tools, streamed) + a small judge alongside (gpt-4.1-mini)
   -> guards, in code  tool locks; claim, promise, language, gender and leak checks on every sentence
   -> voice            ElevenLabs "Sia", sentence by sentence; fixed lines from a cache
 saved notes -> my n8n automation -> WhatsApp
```

## What makes it different

- **Rules enforced in code, not the prompt.** Ringing me needs an agreed offer or an explicit request, a name
  and a reason; my phone rings at most once a call (twice for family) and connects only on my yes. The model
  can't talk past it.
- **Live honesty guards.** Every sentence is checked *before* it is spoken: no "I'm connecting you" while
  nothing rings, no callback promises, no leaked instructions. A blocked sentence is never said.
- **A family flow.** Family numbers ring me straight away; a decline is never re-rung, a no-pickup gets one
  more try, and everything else becomes a reminder or a message in their own words.
- **WhatsApp notes that never slow the call.** An outbox with retries, so a note survives a webhook outage or
  a restart.
- **Evals that check speech, not just actions.** Reply language, promises, claims, leaks, grammar and invented
  facts, on every scripted call, using the same checks as the live guards.

## Read more

- **[What building it taught me](docs/lessons.md)**: seven lessons, each from a real failure.
- [Architecture](docs/architecture.md): authority in code, family flow, WhatsApp notes, known issues.
- [Evals](docs/evals.md): the suite, the honesty guards, grader validation, family scenarios.
- [Latency](docs/latency.md): 15.9 s → 4.3 s step by step, the model bake-off, the judge choice.
- [Speech-to-text and voice](docs/stt-and-voice.md): the bake-off on my own voice, the voice choice.

## Run it

```
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env                                # then fill in your keys (WhatsApp notes and family names are optional)
python prerender.py --render                          # pre-render the fixed lines (ElevenLabs)
python server.py                                      # then open http://localhost:5000
python run_evals.py 3                                 # the eval suite (about $0.25 per full run; --family for the 9 family scenarios)
python test_server.py && python test_streaming.py && python test_notify.py && python test_access.py && python test_provider_down.py   # no API calls
```

To run it publicly, set `JARVIS_ACCESS_CODE` and `JARVIS_SESSION_SECRET` (a lock screen, 12-hour sessions), `JARVIS_DEMO_MODE=1`,
and optionally the usage caps, `JARVIS_BASE_PATH` and `JARVIS_HOST` (see `.env.example`).

Caller audio is never stored: each clip goes from memory to the transcription API and is dropped.
Evals and tests never send WhatsApp notes; only the running server does.
