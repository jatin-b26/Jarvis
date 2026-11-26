# Pre-render every fixed line into tts_cache/, per voice, so it plays with no TTS delay.
#   python prerender.py            show the plan and the ElevenLabs characters it would use (no API calls)
#   python prerender.py --render   render the missing lines
# The cache file name hashes model + voice + settings + text, so a reworded line is simply re-rendered.
import sys
import server
import voice

plan = voice.cache_plan(server.server_fixed_lines())
missing = [(text, lang) for text, lang in plan if voice.cached_pcm(text, lang) is None]
chars = voice.characters_for(plan)
print(f"{len(plan)} lines in the plan ({len({t for t, _ in plan})} distinct texts), {len(missing)} not cached yet.")
print(f"ElevenLabs characters to render the missing ones: {chars} (v4 Turbo bills 0.5x: ~{chars // 2} credits)\n")
for text, lang in plan:
    name = voice.stream_voice(lang)["name"]
    mark = "cached " if voice.cached_pcm(text, lang) is not None else "MISSING"
    print(f"  {mark} {name}: {voice.stream_voice(lang)['prefix']}{text}")

if "--render" in sys.argv and missing:
    rendered, failed = voice.render_cache(plan)
    print(f"\nRendered {len(rendered)}, failed {len(failed)}.")
    for text, name, error in failed:
        print(f"  FAILED {name}: {text[:60]}  ({error})")
