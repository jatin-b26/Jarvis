# Evals and honesty guards

All numbers come from the logged calls (`jarvis.db`, call IDs given) or from eval reports. Run the suite with
`python run_evals.py 3` (about $0.25 per full run) or `python run_evals.py 3 --family` for the family scenarios.

## The eval suite

25 scripted scenarios, 3 calls each (75 calls): callbacks, connect and decline, six family flows, sales and OTP
scams, a rude caller, a caller who changes their mind, Hindi/Hinglish/English and an English-to-Hinglish
switch, Urdu- and Devanagari-script input, a caller asking for the note to be saved again, and small talk. Each
call is checked on **actions** (outcome, tools used and refused, rings) and on **speech**: reply language, no
callback or timing promises, no ringing claims, no leaked instructions, no reading a note back, no ending before
the caller's goodbye, grammar that matches the voice's gender, and invented claims (an LLM grader, itself
validated on 29 labelled replies: 28/29 in each of three runs, every must-flag caught).

Latest full run (#1571-1645): **71/75 as run**. Three of the four failures were my own scenario script (a
"bye" line the call correctly never reached); fixed and re-scored offline, **74/75**. The fourth (a family
caller's message saved twice, as a message and a reminder) led to a prompt rule, and the family-only runs
since are 27/27. Changes made after the last full run (situation-aware fillers, family notes in the caller's
own words, callback-only reminder reasons dropped) are covered by unit tests only. Plus **750 zero-cost unit
checks** (fake model, fake speech, stub webhook).

## Honesty guards: what Jarvis may not say

A voice agent's worst failures are spoken: a promise nobody will keep, or "I'm connecting you" while nothing
rings. Every sentence passes these checks **before** it is spoken. A blocked sentence is never said; the model
gets one retry ("reply again in one short line, then ask your next question"), then code says a fixed line.

| Guard | Blocks | Fallback if the retry fails too |
|---|---|---|
| Claim | Jarvis saying she is ringing, calling, trying or connecting someone, in English or Hinglish ("main ring karti hoon", "please hold"). Only code announces a real ring. | "Should I try calling him, or take a message or set a reminder?" |
| Early offer | An offer to ring before the reason is known | "Could you tell me what it's about first?" (a different wording if that was just said) |
| Promise | Callback or timing promises: "he'll call you back", "I'll ask him to call you back", "soon", "ASAP". Jarvis may say she'll *note* that they want a callback. | as above |
| Language | Hindi words to a caller speaking English | "Sorry, can you say that again?" |
| Gender | Masculine first person ("karta hoon") in the female voice | as above |
| Leak | The model reciting its own instructions (6 words in a row from the prompt or the turn's hint, or self-reference) | as above |

These are pattern nets, not guarantees, so each was run over every past model line before going live: the
claim patterns over 3,673 lines (every hit a real claim); the new promise patterns over 6,749 sentences (14
statements caught, all real promises; 16 callback *offers* phrased as questions correctly left alone). The eval
uses the same functions as the live guard, so live and eval can't disagree. Latest full run (#1571-1645, 75
calls): claims spoken 0, promises spoken 0, leaks spoken 0, masculine forms 0, rings without an agreed offer or
an explicit request 0; the guard stopped 5 early offers.

## Family scenarios

Six eval scenarios cover the family flow ([architecture.md](architecture.md#family-calls)): decline; no pickup
then connected; no pickup twice; a third ring refused; insisting after a decline; "nahi, theek hai" with no
extra ring. Family-only runs, 9 scenarios x3: **27/27** twice (#1646-1669, #1675-1698), 0 ring-limit
violations. In the family run before the message field was tightened, 23 of 23 messages started "Family
called; they asked you..."; in the run after (#1675-1698), none did.
