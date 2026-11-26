# Speech-to-text bake-off on real recordings (2026-11-25). Same clip, every engine, side by side.
#   python stt_bakeoff.py                    Jatin's clips: samples\jatin_recordings\01..08.* vs samples\13\lines.json
#   python stt_bakeoff.py <folder> [--no-ref]  any folder of clips (smoke test)
# Writes samples\13\bakeoff.json and samples\13\bakeoff.md. Clips are read, sent, never modified.
import os
import re
import sys
import json
import time
import glob
import statistics
import httpx
from dotenv import load_dotenv
import voice

load_dotenv()

# Scoring only: Devanagari/Urdu transcripts are rewritten into Roman before counting words wrong. (This was
# the call's display rewrite until Sarvam's Roman output made it unnecessary.)
ROMAN_PROMPT = ("Rewrite this phone-call transcript in Roman script, the way Indians type Hinglish on WhatsApp. "
                "Keep every word: do not translate, summarise, fix or add anything. Write each word once, in one "
                "spelling: never give alternatives and never use '/'. English words written in Devanagari go back "
                "to normal English spelling (domain, renewal, invoice, payment). Output only the rewritten text.")


def has_arabic_script(text):
    return any("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\ufb50" <= ch <= "\ufeff" for ch in text or "")


def has_non_latin(text):
    return any(ch.isalpha() and not ("a" <= ch.lower() <= "z") for ch in text or "")


def roman_for_display(text):
    if not text or not has_non_latin(text):
        return text
    try:
        response = voice.client().chat.completions.create(
            model="gpt-5-nano", reasoning_effort="minimal",
            messages=[{"role": "system", "content": ROMAN_PROMPT}, {"role": "user", "content": text}])
        roman = " ".join((response.choices[0].message.content or "").strip().replace("।", ".").replace("/", " ").split()).strip('"“”')
        return roman if roman and not has_non_latin(roman) else None
    except Exception:
        return None


def old_transcribe(data, name, mime):
    """The call's speech-to-text before 2026-11-25: gpt-4o-mini-transcribe, no prompt, Urdu -> re-run as Hindi."""
    text = (voice.client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=(name, data, mime)).text or "").strip()
    if has_arabic_script(text):
        text = (voice.client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=(name, data, mime),
                                                           language="hi").text or "").strip()
    return text
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples", "13")
RUNS = 2  # each engine twice per clip: transcript from run 1, time = the median of both
USD_PER_MIN = {  # list prices, per audio minute
    "today: gpt-4o-mini-transcribe": 0.003, "gpt-4o-transcribe": 0.006,
    "ElevenLabs Scribe (auto)": 0.20 / 60, "ElevenLabs Scribe (hi+en)": 0.20 / 60,
    "Sarvam saaras:v4 translit": 0.49 / 85, "Sarvam saaras:v4 codemix": 0.49 / 85,  # Rs 0.49/min, ~Rs 85 per USD
}
MIME = {".webm": "audio/webm", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg"}
_http = httpx.Client(timeout=60)
SCRIBE_MODEL = {"id": None}


def openai_engine(model):
    def run(data, name, mime):
        if model == "today":
            return old_transcribe(data, name, mime)  # the call's setup before the bake-off
        return (voice.client().audio.transcriptions.create(model=model, file=(name, data, mime)).text or "").strip()
    return run


def scribe_engine(languages):
    def run(data, name, mime):
        # Newest model first; the docs name Scribe v3 but only show "scribe_v2" as a literal.
        for model in ([SCRIBE_MODEL["id"]] if SCRIBE_MODEL["id"] else ["scribe_v3", "scribe_v2"]):
            fields = {"model_id": model, "tag_audio_events": "false"}
            if languages:
                fields["language_code"] = languages[0]
                if model == "scribe_v3":
                    fields["secondary_language_codes"] = json.dumps(languages[1:])
            r = _http.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY")},
                           data=fields, files={"file": (name, data, mime)})
            if r.status_code == 200:
                SCRIBE_MODEL["id"] = model
                body = r.json()
                ends = [w.get("end") or 0 for w in body.get("words", [])]
                run.duration = max(ends) if ends else None
                return (body.get("text") or "").strip()
            last = f"{model}: {r.status_code} {r.text[:200]}"
        raise RuntimeError(last)
    run.duration = None
    return run


def sarvam_engine(mode):
    def run(data, name, mime):
        r = _http.post("https://api.sarvam.ai/speech-to-text", headers={"api-subscription-key": os.getenv("SARVAM_API_KEY")},
                       data={"model": "saaras:v4", "mode": mode, "language_code": "unknown", "with_timestamps": "true"},
                       files={"file": (name, data, mime)})
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        body = r.json()
        ends = (body.get("timestamps") or {}).get("end_time_seconds") or []
        run.duration = max(ends) if ends else None
        return (body.get("transcript") or "").strip()
    run.duration = None
    return run


scribe_auto, scribe_hi = scribe_engine(None), scribe_engine(["hin", "eng"])
sarvam_translit = sarvam_engine("translit")
ENGINES = [
    ("today: gpt-4o-mini-transcribe", openai_engine("today")),
    ("gpt-4o-transcribe", openai_engine("gpt-4o-transcribe")),
    ("ElevenLabs Scribe (auto)", scribe_auto),
    ("ElevenLabs Scribe (hi+en)", scribe_hi),
    ("Sarvam saaras:v4 translit", sarvam_translit),
    ("Sarvam saaras:v4 codemix", sarvam_engine("codemix")),
]

# Spelling variants that aren't errors in Roman Hinglish.
SAME = {"he": "hai", "hain": "hai", "nahin": "nahi", "hun": "hoon", "hu": "hoon", "hoo": "hoon", "mai": "main",
        "me": "mein", "acha": "accha", "achha": "accha", "achcha": "accha", "thik": "theek", "theek": "theek",
        "ok": "okay", "han": "haan", "ha": "haan", "haa": "haan", "hanji": "haan ji", "bangaluru": "bangalore",
        "bengaluru": "bangalore", "gurugram": "gurgaon", "jayega": "jaega", "jaayega": "jaega", "unko": "unko",
        "karni": "karni", "baare": "bare", "wale": "wale", "vaale": "wale", "waali": "wali", "vali": "wali", "agreemnt": "agreement"}


def words(text):
    out = []
    for w in re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'")):
        out.extend(SAME.get(w, w).split())
    return out


def word_errors(ref, hyp):
    """Substitutions + insertions + deletions (word-level edit distance)."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def script_of(text):
    if has_arabic_script(text):
        return "Urdu"
    if any("ऀ" <= ch <= "ॿ" for ch in text or ""):
        return "Roman+Devanagari" if re.search(r"[A-Za-z]", text) else "Devanagari"
    return "Roman" if text else "-"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(HERE, "samples", "jatin_recordings")
    use_ref = "--no-ref" not in sys.argv
    refs = json.load(open(os.path.join(OUT, "lines.json"), encoding="utf-8")) if use_ref else {}
    clips = sorted(p for p in glob.glob(os.path.join(folder, "*")) if os.path.splitext(p)[1].lower() in MIME)
    if not clips:
        raise SystemExit(f"No clips in {folder}")
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(clips)} clips from {folder}, {len(ENGINES)} engines x {RUNS} runs ({time.strftime('%Y-%m-%d %H:%M')})")
    rows = []
    for path in clips:
        name = os.path.basename(path)
        key = name.split(".")[0].split("_")[0]
        data = open(path, "rb").read()
        mime = MIME[os.path.splitext(name)[1].lower()]
        ref = refs.get(key, "")
        row = {"clip": name, "ref": ref, "engines": {}}
        for label, run in ENGINES:
            texts, times, error = [], [], None
            for _ in range(RUNS):
                started = time.time()
                try:
                    texts.append(run(data, name, mime))
                except Exception as e:
                    error = str(e)[:200]
                    break
                times.append(int((time.time() - started) * 1000))
            text = texts[0] if texts else None
            scored = text
            if text and script_of(text) != "Roman":  # score Devanagari/Urdu after a Roman rewrite (display helper)
                scored = roman_for_display(text) or text
            wrong, total = word_errors(ref, scored) if (ref and text is not None) else (None, len(words(ref)))
            row["engines"][label] = {"text": text, "scored_as": scored if scored != text else None, "script": script_of(text or ""),
                                     "wrong": wrong, "ref_words": total, "ms": int(statistics.median(times)) if times else None,
                                     "same_both_runs": len(set(texts)) == 1 if texts else None, "error": error}
        row["seconds"] = scribe_auto.duration or sarvam_translit.duration  # speech end; trailing silence not counted
        rows.append(row)
        print(f"  {name}: done")
    json.dump(rows, open(os.path.join(OUT, "bakeoff.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_report(rows, folder)


def write_report(rows, folder):
    lines = [f"# STT bake-off ({time.strftime('%Y-%m-%d %H:%M')})", f"Clips: {folder}", ""]
    total_secs = sum(r["seconds"] or 0 for r in rows)
    lines += ["## Summary", "", "| Engine | Words wrong | Median time/clip | Scripts | Cost (these clips) | Cost per hour |", "|---|---|---|---|---|---|"]
    for label, _ in ENGINES:
        cells = [r["engines"][label] for r in rows]
        wrong = sum(c["wrong"] or 0 for c in cells if c["wrong"] is not None)
        ref_words = sum(c["ref_words"] or 0 for c in cells)
        ms = [c["ms"] for c in cells if c["ms"]]
        scripts = ", ".join(sorted({c["script"] for c in cells}))
        errors = sum(1 for c in cells if c["error"])
        lines.append(f"| {label} | {wrong}/{ref_words}" + (f" ({errors} errors)" if errors else "") +
                     f" | {(statistics.median(ms) / 1000 if ms else 0):.2f} s | {scripts} | ${USD_PER_MIN[label] * total_secs / 60:.4f} | ${USD_PER_MIN[label] * 60:.2f} |")
    lines += ["", "## Per clip", ""]
    for r in rows:
        lines += [f"### {r['clip']}" + (f" ({r['seconds']:.1f} s)" if r["seconds"] else ""), f"Said: {r['ref'] or '(no reference)'}", "",
                  "| Engine | Transcript | Script | Wrong | Time |", "|---|---|---|---|---|"]
        for label, _ in ENGINES:
            c = r["engines"][label]
            text = c["error"] and f"ERROR {c['error']}" or (c["text"] or "(empty)")
            if c["scored_as"]:
                text += f" _(scored as: {c['scored_as']})_"
            lines.append(f"| {label} | {text} | {c['script']} | {c['wrong'] if c['wrong'] is not None else '-'} | "
                         f"{(c['ms'] or 0) / 1000:.2f} s{'' if c['same_both_runs'] in (True, None) else ' (2nd run differed)'} |")
        lines.append("")
    open(os.path.join(OUT, "bakeoff.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:4 + 2 + len(ENGINES)]))
    print(f"\nFull report: {os.path.join(OUT, 'bakeoff.md')}")


if __name__ == "__main__":
    main()
