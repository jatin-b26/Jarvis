# Latency

All numbers come from the logged calls (`jarvis.db`, call IDs given) or from eval reports.

## 15.9 s to 4.3 s per turn, like for like

Turn time = model + waiting for the judge + tools (text evals; no speech, no ring), the same 15 original
scenarios:

| Setup | Calls | Median turn | p90 | Worst |
|---|---|---|---|---|
| gpt-5-mini, default reasoning, judge gpt-5-nano | #620-664 | 15.9 s | 25.0 s | 73.8 s |
| reasoning "low" | #679-723 | 7.3 s | 13.1 s | 19.0 s |
| + code-spoken lines + claim guard | #913-957 | 5.5 s | 9.5 s | 20.3 s |
| + judge gpt-4.1-mini, language picked by code | #1141-1185 | 4.0 s | 6.9 s | 16.4 s |
| current (more guards, family flow) | #1571-1615 | **4.3 s** | **7.1 s** | **11.4 s** |

Reasoning effort was the biggest lever, and the cheapest setting broke the agent: "minimal" (#736-780) was
5.0 s but scored 22/57 overall, because it mostly stopped calling tools. The judge used to sit on the voice's
critical path (it chose the reply language); moving that choice to code took the turn's wait for the judge from
a 2.6 s median to 0. OpenAI's speed varies between runs, so treat steps under ~0.5 s as noise.

## Live calls

From releasing the talk button to the first sound of the reply (the page's own clock, filler excluded):
median 10.8 s on #912 (7 turns) before the latency work; 4.7 s and 6.5 s on the two latest live calls (#1509,
5 turns; #1510, 3 turns). Small samples.

A short cached filler ("Achha..." / "One second...") covers slow turns: it waits 2 s, is chosen by situation
(none right after "anything else?", a "working" one after a ring request or a question, an acknowledgement
otherwise), skips about 1 turn in 3 and never plays on two turns in a row: simulated over the last full run, it
played on 44% of turns, and never started with the same word as the reply after it.

## Main-model bake-off: faster models answered in ~1.1 s but broke rules; kept the slower model

Same 19-scenario eval (57 calls) for each candidate, 2026-11-25, 22:54-23:34 IST, with the same judge
(gpt-4.1-mini) and the same guards. Baseline: gpt-5-mini at reasoning "low" (#1141-1197).

| | **gpt-5-mini "low"** (kept) | gpt-4.1-mini | gpt-5.4-mini, no reasoning | Groq gpt-oss-120b "low" |
|---|---|---|---|---|
| Score | **54/57** | 28/57 | 49/57 | 46/57 |
| Tool calls missed | 1 | 20 | 3 | 4 |
| Ringing claims blocked by the guard | 1 | 12 | 0 | 10 |
| Hindi to English callers, blocked by the guard | 0 | 76 | 40 | 23 |
| Fixed "say that again" fallbacks spoken | 0 | 17 | 10 | 3 |
| First sentence ready, median / p90 / worst | 4.1 / 7.1 / 16.4 s | 1.1 / 2.6 / 4.4 s | 1.1 / 2.6 / 4.3 s | 2.8 / 16.5 / 34.4 s |
| Main-model requests per call | 3.54 | ~4.7 | 4.07 | 3.77 |

Why each was rejected:
- **gpt-4.1-mini:** fast because it skipped the work: it never rang in 11 calls, never ended 6, tried to claim
  it was ringing 12 times, and wrote Hindi to English callers 76 times.
- **Groq gpt-oss-120b:** 10 ringing claims stopped by the guard, English replies to Hinglish family callers,
  17 requests failed in Groq's own tool-argument validation, and 240 rate-limit retries on the free tier
  (8,000 tokens per minute) put its p90 at 16.5 s.
- **gpt-5.4-mini (no reasoning): the closest, and a possible future project.** Three narrow, systematic
  failures: it never ended an abusive call (0/3); it wrote Hindi to English callers 40 times (all caught, but
  10 callers then heard a fixed "say that again"); and it wrote 7 saved notes in the third person ("They
  said...", "The caller asked you..."). It made no false claims and was about 3.5x faster.

## The judge choice

The judge was chosen the same way, offline first (`judge_agreement.py`, 539 logged turns): on the two readings
it still makes, gpt-4.1-mini matched the logged verdicts (97.4% on "agreed", 93.9% on "reason known") about as
often as the old judge matched itself on a rerun (95.1%, 93.7%), at a 0.94 s median instead of 5.2 s.
