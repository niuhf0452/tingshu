# Zero-shot voice prompts

Reference audio files consumed by the TTS backend's zero-shot cloning mode
(Qwen3-TTS in production). Engine-independent — the same files work across
any TTS that accepts `(ref_audio, ref_text)`.

## File layout

For each zero-shot speaker referenced in `speakers.json` as
`"speaker_id": "zs:<prompt_id>"`, place two files here:

```
<prompt_id>.wav    # mono or stereo, any sample rate (resampled at load)
<prompt_id>.txt    # the exact text spoken in the wav (verbatim transcript)
```

## Guidelines

- Length: 5–10 seconds of clean speech produces the most natural clones.
- Content: natural Chinese sentences, mixed tones; avoid breathing-only or
  sound-effect clips.
- Speaker: one speaker only — zero-shot clones blend otherwise.
- Text file: transcribe verbatim, including punctuation.

## Populating the library

Two supported paths:

1. **Bulk seeding from Volcengine TTS 2.0 catalog** (current approach):
   run `scripts/fetch_volcengine_voices.py` to generate 46 `volc_*.wav`
   references, then `scripts/build_speakers_from_catalog.py` to
   hand-map tags → attributes in `speakers.json`.

2. **One-off real-audio sources** (movie/podcast clips, real-person
   recordings): drop `<id>.wav` + `<id>.txt` here, add a stub entry in
   `speakers.json`, then run `scripts/annotate_voices.py --ids zs:<id>`
   to let Gemini classify the attributes.

## `.gitignore`

The audio files are binary and host-specific — do not commit them.
`speakers.json` holds the attribute metadata.
