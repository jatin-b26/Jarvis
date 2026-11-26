# Speech-to-text and voice

## Speech-to-text bake-off on my own voice

8 lines recorded in the browser the way I talk on a call (Hinglish with names and places, English, a one-word
"haan", one fast mumbled line), 98 words in all, each clip run twice per setup (`stt_bakeoff.py`). Words wrong
counted by hand; script and number style don't count, meaning does.

| Setup | Words wrong | Script returned | Median time per clip |
|---|---|---|---|
| gpt-4o-mini-transcribe (previous setup) | ~24 | Devanagari, Urdu, **Malayalam**, Roman | 0.88 s |
| gpt-4o-transcribe | ~15 | Devanagari, Urdu, Gurmukhi, Roman | 1.02 s |
| ElevenLabs Scribe v3 (auto / Hindi+English) | 3 / 3 | Devanagari with English words in Roman | 0.70 s |
| **Sarvam saaras:v4, Roman output** | **2** | **Roman, every clip** | **0.41 s** |
| Sarvam saaras:v4, code-mix output | 2 | Devanagari with English words in Roman | 0.36 s |

Only Sarvam heard "Can **he** call me back" (a callback request) rather than "Can **you**". Caveat: 8 clips, one
voice (mine), a quiet room: enough to rule out a transcriber that returns the wrong script, not to rank two
close ones for every caller. On live calls since, Sarvam took 0.27-0.52 s per clip (#1509, #1510).

## Voice

Five text-to-speech options were compared by ear and by time to first audio, measured from India: OpenAI
(robotic), ElevenLabs, Sarvam (two models) and Google Chirp 3 HD. ElevenLabs' Indian-accented "Sia" voice was
picked by ear. Measured on the live path: pre-opening the connection at call start cut the first streamed line
from 675 ms to 184 ms; streamed lines on #1509 started in 243-1,106 ms (median 428 ms); fixed lines play from a
pre-rendered cache with no synthesis at all.
