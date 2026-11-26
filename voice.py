# Speech for Jarvis: speech-to-text (Sarvam saaras:v4 Roman, ElevenLabs Scribe as the fallback) and
# text-to-speech (ElevenLabs Sia; the Sarvam voices below are legacy).
#
# Text-to-speech uses Sarvam bulbul:v4-flash over a WebSocket, kept open for the whole call, so each
# reply starts arriving in ~200 ms (measured warm). Two voices, picked by the language the judge
# reports each turn: sunny for English, aayan for Hinglish. SARVAM_API_KEY must be in .env.
# Optional (off by default): TTS_VOICE_SET=elevenlabs_sia uses ElevenLabs (female), with a female
# Sarvam voice as the fallback. See VOICE_SETS.
import os
import re
import json
import time
import base64
import httpx  # already installed with the openai SDK (not a new dependency)
from dotenv import load_dotenv
from openai import OpenAI
from websockets.sync.client import connect

load_dotenv()

# ---------- speech-to-text ----------
# Bake-off on Jatin's own voice (2026-11-25, samples\13\bakeoff.md): Sarvam saaras:v4 in "translit" mode got
# the most words right (2 of 98 wrong), is the fastest (0.41 s median per clip) and is the only setup that
# always returns Roman script. gpt-4o-mini-transcribe (used until now) returned Urdu and even Malayalam.
# So: Sarvam translit first. ElevenLabs Scribe v3 (Hindi + English, second best: 3 wrong, Devanagari with
# English words in Roman) when Sarvam fails, and for turns over 30 s (Sarvam's real-time limit; no splitting).
# Scribe's Devanagari reaches the model as is: agent.in_indic_script() treats such a line as Hinglish.
# No prompt or hint words are sent to either: the old hint came back as a transcript (live call 2026-11-25).
# Caller audio is never saved: the bytes go straight to the API and are then dropped.

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SARVAM_MAX_SECONDS = 30
last_stt = {"engine": None, "why": None}  # which engine transcribed the last turn, and why the fallback ran
_stt_http = {"client": None}


def _stt_client():
    if _stt_http["client"] is None:
        _stt_http["client"] = httpx.Client(timeout=httpx.Timeout(15, connect=5))  # kept alive across turns
    return _stt_http["client"]


def warm_stt():
    """Open the kept-alive Sarvam connection before the first turn (saves the TLS handshake). Never raises."""
    try:
        _stt_client().head("https://api.sarvam.ai/")
    except Exception:
        pass


def _sarvam(audio_bytes, filename, content_type):
    r = _stt_client().post(SARVAM_STT_URL, headers={"api-subscription-key": os.getenv("SARVAM_API_KEY") or ""},
                           data={"model": "saaras:v4", "mode": "translit", "language_code": "unknown"},
                           files={"file": (filename, audio_bytes, content_type)})  # straight from memory
    if r.status_code != 200:
        raise RuntimeError(f"Sarvam {r.status_code}")
    return (r.json().get("transcript") or "").strip()


def _scribe(audio_bytes, filename, content_type):
    r = _stt_client().post(SCRIBE_URL, headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY") or ""},
                           data={"model_id": "scribe_v3", "language_code": "hin",
                                 "secondary_language_codes": json.dumps(["eng"]), "tag_audio_events": "false"},
                           files={"file": (filename, audio_bytes, content_type)}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Scribe {r.status_code}")
    return (r.json().get("text") or "").strip()


def transcribe(audio_bytes, filename="turn.webm", content_type="audio/webm", seconds=None):
    """Turn one clip of caller audio into text. seconds: the clip's length, from the page (None = unknown).
    Returns (text, milliseconds). text is None if both engines failed. Never raises.
    last_stt says which engine it was ("sarvam" / "scribe") and, for Scribe, why ("over 30 s" / "Sarvam ...")."""
    started = time.time()
    last_stt.update(engine=None, why=None)
    if seconds is not None and seconds > SARVAM_MAX_SECONDS:
        last_stt["why"] = f"over {SARVAM_MAX_SECONDS} s"
    else:
        try:
            text = _sarvam(audio_bytes, filename, content_type)
            last_stt["engine"] = "sarvam"
            return text, int((time.time() - started) * 1000)
        except Exception as error:
            last_stt["why"] = f"Sarvam failed: {str(error)[:60]}"
    try:
        text = _scribe(audio_bytes, filename, content_type)
        last_stt["engine"] = "scribe"
    except Exception as error:
        last_stt["why"] = (last_stt["why"] or "") + f"; Scribe failed: {str(error)[:60]}"
        text = None
    return text, int((time.time() - started) * 1000)


# A transcript of only these (or of nothing but punctuation) is not speech to answer: "didn't catch that".
# Non-words only: "haan", "ok", "bye", "thank you" are real answers and pass through.
FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "hmm", "hmmm", "hm", "mm", "mmm", "mhm", "ah", "aah", "ahh", "er",
                "erm", "eh", "huh", "हम्म", "हम", "अं", "उम", "उम्म", "आह", "ہم", "ہمم"}
# The old hint, in case a model returns it anyway: a transcript made only of these words is an echo.
OLD_HINT_WORDS = {"jarvis", "jatin", "phone", "call", "english", "hindi", "hinglish"}


def not_speech(text):
    """True if a transcript has nothing to answer: empty, only filler or punctuation, or the old hint echoed."""
    words = re.findall(r"[^\s.,!?;:'\"“”‘’…()\-–—।]+", (text or "").lower())  # not \w: it splits Devanagari
    if not words or all(w in FILLER_WORDS for w in words):
        return True
    return len(words) >= 4 and set(words) <= OLD_HINT_WORDS


_client = None


def client():
    """The OpenAI client (stt_bakeoff.py's OpenAI setups). No hidden SDK retries."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
    return _client


# ---------- text-to-speech ----------

# send_completion_event=true makes Sarvam send a "final" event after the last audio chunk,
# so we know the clip is complete without waiting on a timer.
TTS_URL = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v4-flash&send_completion_event=true"
VOICES = {  # judge's language -> (Sarvam speaker, language code). Chosen by ear in the bake-off (samples\7, 8).
    "english": ("sunny_enhi_customer", "en-IN"),
    "hinglish": ("aayan_hi_conversational", "hi-IN"),
}

# Voice sets. Every voice in a set has ONE gender, because Hinglish first-person grammar follows it
# ("main karta hoon" / "main karti hoon"). TTS_VOICE_SET in .env picks the set.
# Default "sia" (Jatin, 2026-11-24, after the live test): ONE ElevenLabs voice, Sia 05, for both languages
# and the opener. (Two voices, 06 + 05, sounded like two people.) "sarvam" stays for the legacy speak() only.
SIA05 = "ivYuatsaZtRtKoHJGZY3"  # "Sia - Confident Auto Care Voice"
VOICE_SETS = {
    "sarvam": {"provider": "sarvam", "gender": "male", "voices": VOICES},
    "sia": {
        "provider": "elevenlabs", "gender": "female",
        "voices": {"english": SIA05, "hinglish": SIA05},
        # No backup voice: if ElevenLabs fails, server.py plays the cached outage line, saves the note and
        # ends the call. The female Sarvam fallback stays in speak(), switched OFF.
        "fallback": {"english": ("simran_enhi_customer", "en-IN"), "hinglish": ("simran_enhi_customer", "hi-IN")},
    },
}
SARVAM_FALLBACK_ON = False
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo")

# How Sia 05 says each language. English keeps the accent fixes (Jatin's pass 5 pick): the tag, similarity
# 1.0, stability 0.75. The tag goes on ENGLISH text only. Hinglish: same voice settings, no tag
# (Jatin picked clip 2, samples\12\tag, 2026-11-24). Gain -0.7 dB: Sia 05 measured -20.3 dBFS, target -21.0.
ACCENT_SETTINGS = {"stability": 0.75, "similarity_boost": 1.0, "style": 0.0, "use_speaker_boost": True, "speed": 1.0}
SIA = {
    "english": {"name": "sia05", "id": SIA05, "prefix": "[Indian English accent] ", "settings": ACCENT_SETTINGS, "gain_db": -0.7},
    "hinglish": {"name": "sia05", "id": SIA05, "prefix": "", "settings": ACCENT_SETTINGS, "gain_db": -0.7},
}


def voice_set():
    return VOICE_SETS.get(os.getenv("TTS_VOICE_SET", "sia"), VOICE_SETS["sia"])


def gender():
    """The active voice's gender: "male" or "female". Jarvis's Hinglish grammar follows it."""
    return voice_set()["gender"]


TTS_PACE = 0.95          # 0.9 sounded worst; 1.0 slightly fast
TTS_DICTIONARY = "p_08d44334"  # pronunciation fixes: theek, maine, mai/main, koi, sure, gets

_tts = {"ws": None, "voice": None, "http": None, "fallback": False}  # one open connection per call


def _use_voice(lang, voices=VOICES):
    """Open the connection if needed, and switch voice if the language changed. A new config
    message switches voice on the same connection (tested), so no reconnect is needed."""
    speaker, code = voices.get(lang, voices["english"])  # language unset or unknown -> English voice
    if _tts["ws"] is None:
        _tts["ws"] = connect(TTS_URL, additional_headers={"api-subscription-key": os.getenv("SARVAM_API_KEY")},
                             max_size=None, open_timeout=5)
        _tts["voice"] = None
    if _tts["voice"] != speaker:
        _tts["ws"].send(json.dumps({"type": "config", "data": {
            "speaker": speaker, "target_language_code": code, "output_audio_codec": "mp3",
            "speech_sample_rate": 24000, "pace": TTS_PACE, "dict_id": TTS_DICTIONARY}}))
        _tts["voice"] = speaker
    return speaker


def speak(text, lang="english"):
    """Turn one reply into speech. Returns (mp3 bytes, speaker, ms to first audio, ms total).
    Audio is None if it failed twice. Never raises: the caller still gets the text."""
    chosen = voice_set()
    if chosen["provider"] == "elevenlabs" and not _tts["fallback"]:
        result = _speak_elevenlabs(text, chosen["voices"].get(lang, chosen["voices"]["english"]))
        if result[0] is not None:
            return result
        if not SARVAM_FALLBACK_ON:
            return None, None, None, None
        _tts["fallback"] = True  # sticky for the rest of the call: one voice change at most, same gender
    return _speak_sarvam(text, lang, chosen["fallback"] if chosen["provider"] == "elevenlabs" else chosen["voices"])


def _speak_elevenlabs(text, voice_id):
    """ElevenLabs over HTTP streaming, one kept-alive connection per call. Same return shape as speak()."""
    for attempt in (1, 2):
        started = time.time()
        try:
            if _tts["http"] is None:
                _tts["http"] = httpx.Client(timeout=15)
            audio, first = b"", None
            with _tts["http"].stream("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                                     headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY") or ""},
                                     params={"output_format": "mp3_44100_128"},
                                     json={"text": text, "model_id": ELEVENLABS_MODEL}) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"ElevenLabs {response.status_code}")
                for chunk in response.iter_bytes():
                    if chunk and first is None:
                        first = time.time() - started
                    audio += chunk
            total = time.time() - started
            return audio, voice_id, int((first or total) * 1000), int(total * 1000)
        except Exception:
            _close_http()
    return None, None, None, None


def _speak_sarvam(text, lang, voices):
    for attempt in (1, 2):
        started = time.time()
        try:
            speaker = _use_voice(lang, voices)
            ws = _tts["ws"]
            ws.send(json.dumps({"type": "text", "data": {"text": text}}))
            ws.send(json.dumps({"type": "flush"}))
            audio, first = b"", None
            while True:
                message = json.loads(ws.recv(timeout=10))
                if message.get("type") == "audio":
                    if first is None:
                        first = time.time() - started
                    audio += base64.b64decode(message["data"]["audio"])
                elif message.get("type") == "event":  # the "final" event: clip complete
                    break
                elif message.get("type") == "error":
                    raise RuntimeError(str(message)[:200])
            total = time.time() - started
            return audio, speaker, int((first or total) * 1000), int(total * 1000)
        except Exception:
            close_tts()  # drop the broken connection; the second attempt opens a fresh one
    return None, None, None, None


def _close_http():
    try:
        if _tts["http"] is not None:
            _tts["http"].close()
    except Exception:
        pass
    _tts["http"] = None


def close_tts():
    """Close the call's TTS connections. Safe to call any time. A new call starts on the primary voice."""
    try:
        if _tts["ws"] is not None:
            _tts["ws"].close()
    except Exception:
        pass
    _close_http()
    _tts["ws"], _tts["voice"], _tts["fallback"] = None, None, False


# ---------- Sia: streamed PCM and the fixed-line cache (step A, 2026-11-24) ----------
# Replies stream as 24 kHz 16-bit mono PCM: the browser plays each chunk as it arrives (Web Audio), so the
# caller waits for the FIRST audio, not the whole clip. Fixed lines (openers, follow-ups, closing lines,
# hold line, fallbacks) are rendered once into tts_cache/ and play with no TTS call at all.
import hashlib
import wave
import audio_level

PCM_RATE = 24000
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
_http = {"client": None}  # kept alive across calls: a warm connection saves ~1 s on the first line


def stream_voice(lang):
    """How Sia 05 says a line in this language (English gets the accent tag, Hinglish doesn't)."""
    return SIA.get(lang or "english", SIA["english"])


def _body(text, v):
    body = {"text": v["prefix"] + text, "model_id": ELEVENLABS_MODEL}
    if v["settings"]:
        body["voice_settings"] = v["settings"]
    return body


def cache_path(text, lang):
    """Where this line lives in this voice. The name hashes everything that changes the audio (model,
    voice, settings, prefix, text), so a reworded line or a new setting is a cache miss, never a stale hit."""
    v = stream_voice(lang)
    key = json.dumps([ELEVENLABS_MODEL, v["id"], v["settings"], v["prefix"], text], sort_keys=True)
    return os.path.join(CACHE_DIR, v["name"], hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".wav")


def cached_pcm(text, lang):
    """The line's PCM from the cache, levelled for its voice, or None if it isn't cached."""
    path = cache_path(text, lang)
    if not os.path.exists(path):
        return None
    with wave.open(path, "rb") as clip:
        pcm = clip.readframes(clip.getnframes())
    return audio_level.apply_gain(pcm, stream_voice(lang)["gain_db"])


def _el_client():
    if _http["client"] is None:
        _http["client"] = httpx.Client(timeout=httpx.Timeout(15, connect=5))
    return _http["client"]


def warm_tts():
    """Open the kept-alive ElevenLabs connection ahead of the first streamed line. Never raises."""
    try:
        _el_client().get("https://api.elevenlabs.io/v1/models", headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY") or ""})
    except Exception:
        pass


def open_stream(text, lang):
    """Start streaming one line. Returns (chunks, voice name, ms to first audio) once the FIRST chunk has
    arrived, or None if ElevenLabs failed twice before any audio. Never raises."""
    v = stream_voice(lang)
    for attempt in (1, 2):
        started = time.time()
        try:
            client = _el_client()
            request = client.build_request(
                "POST", f"https://api.elevenlabs.io/v1/text-to-speech/{v['id']}/stream",
                headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY") or ""},
                params={"output_format": f"pcm_{PCM_RATE}"}, json=_body(text, v))
            response = client.send(request, stream=True)
            if response.status_code != 200:
                response.close()
                raise RuntimeError(f"ElevenLabs {response.status_code}")
            pieces = response.iter_bytes()
            first = next(chunk for chunk in pieces if chunk)
            first_ms = int((time.time() - started) * 1000)
            return _levelled(first, pieces, response, v["gain_db"]), v["name"], first_ms
        except Exception:
            try:
                _http["client"].close()
            except Exception:
                pass
            _http["client"] = None
    return None


def _levelled(first, pieces, response, gain_db):
    """Yield whole 16-bit samples with the voice's gain applied (a network chunk can end mid-sample)."""
    left = b""
    try:
        for chunk in _chain(first, pieces):
            data = left + chunk
            cut = len(data) // 2 * 2
            left = data[cut:]
            if cut:
                yield audio_level.apply_gain(data[:cut], gain_db)
    finally:
        response.close()


def _chain(first, pieces):
    yield first
    for chunk in pieces:
        if chunk:
            yield chunk


def cache_plan(lines_by_lang):
    """Which (text, language) pairs the cache needs: every fixed line in its own language, one voice."""
    return [(text, lang) for lang in ("english", "hinglish") for text in sorted(lines_by_lang[lang])]


def characters_for(plan):
    """ElevenLabs characters the plan sends (before the 0.5x v4 Turbo credit rate), skipping cached lines."""
    return sum(len(stream_voice(lang)["prefix"] + text) for text, lang in plan
               if not os.path.exists(cache_path(text, lang)))


def render_cache(plan):
    """Render every missing line into tts_cache/. Returns (rendered, failed) lists."""
    rendered, failed = [], []
    for text, lang in plan:
        path = cache_path(text, lang)
        if os.path.exists(path):
            continue
        v = stream_voice(lang)
        try:
            response = _el_client().post(f"https://api.elevenlabs.io/v1/text-to-speech/{v['id']}",
                                      headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY") or ""},
                                      params={"output_format": f"pcm_{PCM_RATE}"}, json=_body(text, v), timeout=60)
            if response.status_code != 200:
                raise RuntimeError(f"ElevenLabs {response.status_code}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with wave.open(path, "wb") as clip:
                clip.setnchannels(1)
                clip.setsampwidth(2)
                clip.setframerate(PCM_RATE)
                clip.writeframes(response.content)
            rendered.append((text, v["name"]))
        except Exception as error:
            failed.append((text, v["name"], str(error)[:80]))
    return rendered, failed
