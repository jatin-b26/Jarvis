# Loudness and pace of 16-bit mono PCM, with the standard library only (no ffmpeg, no numpy).
# Used to match Sia 05 and Sia 06 so the switch between them isn't jarring, and to level cached lines.
import array
import math

SAMPLE_RATE = 24000
FRAME = SAMPLE_RATE // 50          # 20 ms frames
SPEECH_FLOOR_DB = -45.0            # a frame quieter than this counts as silence


def samples(pcm_bytes):
    data = array.array("h")
    data.frombytes(pcm_bytes[: len(pcm_bytes) // 2 * 2])
    return data


def _frame_db(frame):
    if not frame:
        return -120.0
    rms = math.sqrt(sum(x * x for x in frame) / len(frame))
    return 20 * math.log10(rms / 32768) if rms > 0 else -120.0


def measure(pcm_bytes):
    """(loudness of the spoken parts in dBFS, seconds of speech, total seconds)."""
    data = samples(pcm_bytes)
    voiced, energy = 0, 0.0
    for start in range(0, len(data), FRAME):
        frame = data[start:start + FRAME]
        if _frame_db(frame) > SPEECH_FLOOR_DB:
            voiced += len(frame)
            energy += sum(x * x for x in frame)
    loud = 20 * math.log10(math.sqrt(energy / voiced) / 32768) if voiced else -120.0
    return loud, voiced / SAMPLE_RATE, len(data) / SAMPLE_RATE


def apply_gain(pcm_bytes, gain_db):
    """Scale 16-bit PCM by gain_db, clipping at full scale."""
    if not gain_db:
        return pcm_bytes
    factor = 10 ** (gain_db / 20)
    data = samples(pcm_bytes)
    for i, x in enumerate(data):
        data[i] = max(-32768, min(32767, int(x * factor)))
    return data.tobytes()
