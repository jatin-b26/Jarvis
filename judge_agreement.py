# Offline check (latency step 1): would gpt-5-nano at MINIMAL reasoning give the same judge verdicts as the
# logged ones (default reasoning)? Replays each logged turn's transcript through the minimal judge, applies the
# same code rules the call applies (script/Hindi-word language override, "agreed" only after caller_wants_jatin,
# reason_known accumulates), and compares with what the call logged.
#   python judge_agreement.py <first call id> <last call id>
import sys, json, time, statistics
from concurrent.futures import ThreadPoolExecutor
import agent, db

FIRST, LAST = int(sys.argv[1]), int(sys.argv[2])
EFFORT = sys.argv[3] if len(sys.argv) > 3 else "minimal"   # a gpt-5 reasoning effort, or "none" for a non-reasoning model
MODEL = sys.argv[4] if len(sys.argv) > 4 else agent.JUDGE_MODEL
OPTIONS = {} if EFFORT == "none" else {"reasoning_effort": EFFORT}
fixed_hold = "(hold line)"


def turns_of(call_id):
    """One entry per judge event: the transcript the judge saw, the caller line, whether caller_wants_jatin had
    fired before it, and the logged (applied) verdict and judge time."""
    history, out, wanted, timings = [], [], False, []
    current = None
    for row in db.replay(call_id):
        kind, d = row["kind"], row["detail"]
        if kind == "caller_said":
            history.append(("Caller", d))
            current = {"call": call_id, "transcript": "\n".join(f"{w}: {t}" for w, t in history), "caller": d}
        elif kind == "jarvis_said" and not d.startswith(fixed_hold):
            history.append(("Jarvis", d))
        elif kind == "tool_called" and d.startswith("caller_wants_jatin"):
            wanted = True
        elif kind == "judge" and d.startswith("lang=") and current is not None and "logged" not in current:
            f = dict(p.split("=") for p in d.split())
            current.update(logged={"lang": f["lang"], "agreed": f["agreed"] == "True", "reason": f["reason_known"] == "True"},
                           wanted=wanted)
            out.append(current)
        elif kind == "timing" and out:
            out[-1].setdefault("judge_ms", json.loads(d)["judge"])
    return out


def ask(transcript):
    started = time.time()
    try:
        r = agent.client.chat.completions.create(
            model=MODEL, **OPTIONS, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": agent.JUDGE_PROMPT}, {"role": "user", "content": transcript}])
        return json.loads(r.choices[0].message.content), int((time.time() - started) * 1000), r.usage.completion_tokens
    except Exception as error:
        return {"failed": str(error)[:80]}, int((time.time() - started) * 1000), 0


calls = [c for c in range(FIRST, LAST + 1)]
all_turns = [t for c in calls for t in turns_of(c)]
with ThreadPoolExecutor(max_workers=6) as pool:
    answers = list(pool.map(lambda t: ask(t["transcript"]), all_turns))

agree = {"lang": 0, "agreed": 0, "reason": 0}
disagreements, ms_new, ms_old, out_tokens, failed = [], [], [], [], 0
reason_so_far = {}
for t, (raw, ms, out) in zip(all_turns, answers):
    if "failed" in raw:
        failed += 1
        continue
    lang = "hinglish" if raw.get("language") == "hinglish" or agent.in_indic_script(t["caller"]) or agent.has_hindi_words(t["caller"]) else "english"
    agreed = bool(raw.get("caller_agreed_to_connect")) if t["wanted"] else False
    reason = reason_so_far.get(t["call"], False) or bool(raw.get("reason_known"))
    reason_so_far[t["call"]] = reason
    new = {"lang": lang, "agreed": agreed, "reason": reason}
    for k in agree:
        if new[k] == t["logged"][k]:
            agree[k] += 1
        else:
            disagreements.append((t["call"], k, t["logged"][k], new[k], t["caller"]))
    ms_new.append(ms)
    out_tokens.append(out)
    if t.get("judge_ms"):
        ms_old.append(t["judge_ms"])

n = len(all_turns) - failed


def spread(xs):
    xs = sorted(xs)
    return f"median {statistics.median(xs) / 1000:.2f} s, p90 {xs[int(len(xs) * 0.9)] / 1000:.2f} s, worst {xs[-1] / 1000:.2f} s"


print(f"calls #{FIRST}-{LAST}: {len(all_turns)} judged turns, {failed} failed requests")
for k, v in agree.items():
    print(f"  {k:7}: {v}/{n} agree ({100 * v / n:.1f}%)")
print(f"  all three: {n - len({(d[0], d[4]) for d in disagreements})}/{n} turns fully agree")
print(f"judge time, logged (default reasoning): {spread(ms_old)}  (n={len(ms_old)})")
print(f"judge time, {MODEL} {EFFORT} (this run, 6 in parallel): {spread(ms_new)}; output tokens median {statistics.median(out_tokens)}")
print(f"\ndisagreements ({len(disagreements)}): call | field | logged -> minimal | caller line")
for d in disagreements:
    print(f"  #{d[0]} | {d[1]} | {d[2]} -> {d[3]} | {d[4][:80]}")
