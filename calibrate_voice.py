"""
calibrate_voice.py  --  helps you choose VOICE_THRESHOLD in app.py

Run from the repo root:
    python calibrate_voice.py

It runs Member 3's voice model on:
  * every recording in audio/<member_id>/*.wav       (GENUINE voices)
  * every .wav directly inside samples/              (STRANGER voices, e.g. samples/unknown_voice.wav)
and prints the confidence for each, then suggests a threshold.

Note: the genuine recordings in audio/ were used to TRAIN the model, so their scores
are optimistic. Record 1-2 NEW clips per member and keep them in
samples/new_<member_id>_yes_approve.wav  (e.g. samples/new_member_01_yes_approve.wav).
Those are scored as genuine, but fairly (the model never saw them).
"""

from pathlib import Path

import app  # uses the same voice code as the real app

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
SAMPLES_DIR = ROOT / "samples"


def score(path):
    label, conf = app._member3_voice(path)
    return label, conf


def main():
    genuine, strangers = [], []

    print("=== GENUINE recordings (audio/<member_id>/) - used in training, so optimistic ===")
    for folder in sorted(p for p in AUDIO_DIR.iterdir() if p.is_dir()):
        for wav in sorted(folder.glob("*.wav")):
            label, conf = score(wav)
            ok = label == folder.name
            genuine.append(conf if ok else 0.0)
            print(f"  {folder.name}/{wav.name:<28} -> {label}  {conf:.2f}  {'OK' if ok else 'WRONG PERSON'}")

    new_clips, stranger_clips = [], []
    if SAMPLES_DIR.exists():
        for wav in sorted(SAMPLES_DIR.glob("*.wav")):
            (new_clips if wav.name.startswith("new_member_") else stranger_clips).append(wav)

    if new_clips:
        print("\n=== NEW genuine recordings (samples/new_member_XX_*.wav) - fair test ===")
        for wav in new_clips:
            expected = "_".join(wav.stem.split("_")[1:3])  # new_member_01_... -> member_01
            label, conf = score(wav)
            ok = label == expected
            genuine.append(conf if ok else 0.0)
            print(f"  {wav.name:<36} -> {label}  {conf:.2f}  {'OK' if ok else 'WRONG PERSON'}")

    if stranger_clips:
        print("\n=== STRANGER recordings (samples/*.wav) - should be rejected ===")
        for wav in stranger_clips:
            label, conf = score(wav)
            strangers.append(conf)
            print(f"  {wav.name:<36} -> best guess {label}  {conf:.2f}")

    print("\n=== SUMMARY ===")
    if genuine:
        print(f"  lowest genuine confidence : {min(genuine):.2f}")
    if strangers:
        print(f"  highest stranger confidence: {max(strangers):.2f}")
    if genuine and strangers:
        lo, hi = min(genuine), max(strangers)
        if lo > hi:
            print(f"  -> clean gap. Suggested VOICE_THRESHOLD = {(lo + hi) / 2:.2f}")
        else:
            print("  -> genuine and stranger scores OVERLAP: no threshold separates them perfectly.")
            print("     The model needs more / augmented recordings from Member 3. For the demo, "
                  f"pick {hi + 0.01:.2f} to reject your stranger clip(s) (some genuine clips may then be denied).")
    else:
        print("  Add a stranger clip as samples/unknown_voice.wav to get a threshold suggestion.")


if __name__ == "__main__":
    main()