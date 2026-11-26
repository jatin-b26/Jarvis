# Replay one call from the audit log:   python show.py <call id>
# Timing lines (step 6) show where each turn's time went, stage by stage, in milliseconds.
import sys
import json
import statistics
import db
import notify

# Windows consoles default to cp1252, which can't print "—" or emoji: print what it can instead of crashing.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

call_id = int(sys.argv[1])


def timing_line(t):
    """One turn's timing as one readable line. Voice stages appear only on web calls."""
    parts = []
    if "stt" in t:
        parts.append(f"stt {t['stt']}")
    if "judge_wait" in t:  # Week 3: the judge runs alongside the model; only the wait costs the caller
        parts.append(f"judge {t['judge']} (turn waited {t['judge_wait']})")
    else:
        parts.append(f"judge {t['judge']}")
    calls = t["model"]
    detail = f" ({len(calls)} calls: {'+'.join(map(str, calls))})" if len(calls) > 1 else ""
    parts.append(f"model {sum(calls)}{detail}")
    parts.append(f"tools {t['tools']}")
    if t["ring"]:
        parts.append(f"ring {t['ring']}")
    if "tts_first" in t:
        parts.append(f"tts {t['tts_first']} first / {t['tts_total']} total")
    line = f"turn {t['turn']}: " + " | ".join(parts)
    if "waited" in t:
        line += f"  => caller waited {t['waited'] / 1000:.1f}s for first audio"
    return line


turns, speech = [], []
for row in db.replay(call_id):
    detail = row["detail"]
    if row["kind"] == "timing":
        t = json.loads(detail)
        turns.append(t)
        detail = timing_line(t)
    elif row["kind"] == "client_timing":  # the page's clock: release -> reply text, release -> first sound
        ct = json.loads(detail)
        detail = (f"turn {ct['turn']}: page saw the reply text {ct.get('to_reply_text', 0) / 1000:.1f}s and the first "
                  f"sound {ct.get('to_first_sound', 0) / 1000:.1f}s after the talk button was released")
    elif row["kind"] == "speech":  # step A: one line played by /speech (cached or streamed)
        sp = json.loads(detail)
        speech.append(sp)
        if sp.get("failed"):
            detail = f"turn {sp['turn']}: {sp['voice']} FAILED, outage line played"
        else:
            how = "cached" if sp["cached"] else f"streamed, first audio {sp['first']} ms, done {sp.get('total')} ms"
            detail = f"turn {sp['turn']}: {sp['voice']} {how}  => caller waited {sp['waited'] / 1000:.1f}s for first audio"
    print(f"{row['at']}  {row['kind']:<15} {detail}")

if turns:
    # Medians across this call's turns: the numbers to quote. Ring is left out (it's Jatin, not the system).
    stages = {"stt": "stt", "judge": "judge", "judge_wait": "judge wait", "model": "model", "tools": "tools",
              "tts_first": "tts first audio", "waited": "caller waited"}
    medians = []
    for key, label in stages.items():
        values = [sum(t[key]) if key == "model" else t[key] for t in turns if key in t]
        if values:
            medians.append(f"{label} {statistics.median(values):.0f}")
    streamed = [sp for sp in speech if not sp.get("failed") and not sp["cached"]]
    if streamed:
        medians.append(f"tts first audio (streamed) {statistics.median(sp['first'] for sp in streamed):.0f}")
    played = [sp for sp in speech if not sp.get("failed")]
    if played:
        medians.append(f"caller waited (speech) {statistics.median(sp['waited'] for sp in played):.0f}")
    print(f"\nStage medians over {len(turns)} turns (ms): " + " | ".join(medians))

left = notify.pending(call_id)
if left:  # phase 3: notes for this call not delivered to WhatsApp yet (or given up)
    print("\nWhatsApp notes not delivered:")
    for row in left:
        print(f"  {row['note_id']}: {row['status']}, {row['attempts']} attempt(s), last error: {row['last_error']}")

print("\nOpen reminders:")
for row in db.open_reminders():
    print(f"  #{row['id']} [{row['kind']}] {row['caller_name']}: {row['content']}")
