# What building Jarvis taught me

Seven lessons, each from a real failure. Call numbers (#) refer to logged calls; every call can be replayed
with `python show.py <call id>`.

1. **An eval measures only what it was built to measure.** Twice a perfect score hid real rule-breaking,
   because the checks looked at *actions* (which tools ran) and not *speech* (what Jarvis said). The suite now
   checks speech on every call.
2. **My scripted callers encoded my assumptions (#1510).** Every eval caller asked "can I talk to Jatin?", so a
   rule that offers needed that request scored fine for weeks. The first live caller who simply gave a name and
   a reason had every offer blocked and hung up. Real callers dial my number: wanting me is implied. Now name +
   reason is the request, and the eval has callers who never ask.
3. **Fields that record vs fields that predict.** A tool field asking the model to predict that the caller was
   finished scored 44/45 while 15 of 45 calls ended before the caller said bye. A field that records what was
   said is fine; one that asks the model to guess what happens next is a decision in disguise. It's gone.
4. **Naturalness is invisible to rule checks.** The tool result said "offer to try him", and the model said
   "Main Jatin ko try karun?" on 30 of 36 Hinglish offers, a literal translation no one says, while every eval
   passed. In a second language the model translates instructions literally, so write the instruction in the
   shape you want spoken.
5. **A check that scores is not a check that enforces (#912).** The eval had a "claimed to be connecting"
   check, but nothing stopped the claim on a live call. The ring was correctly refused, and Jarvis told the
   caller "main Jatin ko abhi ring karti hoon" twice while nothing rang. The same patterns now run live on every
   sentence, and the eval uses them too.
6. **Remove the option instead of instructing against it.** The most reliable fixes took a choice away from
   the model: the ring tool hidden until it can be used, fixed lines for moments with nothing to decide, the
   reply language picked by code.
7. **Measure the path you'll deploy.** ElevenLabs publishes about 75 ms to first audio for Flash v2.5; from
   my machine in India I measured 521 ms (median of 3 runs), because the published figure leaves out the network.
   Sarvam's streaming came close to its claim (under 250 ms published, about 280 ms measured over a kept-open
   WebSocket).
