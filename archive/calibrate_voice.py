"""
calibrate_voice.py  --  helps you choose VOICE_THRESHOLD in app.py

Run from the repo root:   python calibrate_voice.py

Scores the voice model chosen by VOICE_BACKEND in app.py on:
  * every recording in audio/<member_id>/            GENUINE (used in training, so optimistic)
  * every recording in audio/unknown/                STRANGER (not a team member - must be rejected)
  * optional: samples/new_member_01_*.wav ...        NEW genuine clips (fair test, never trained on)
Then it prints a suggested VOICE_THRESHOLD.
"""
from pathlib import Path

import app

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
SAMPLES_DIR = ROOT / "samples"
IGNORED = {"unknown"}


def score(path):
    return app.predict_voice_backend(path)     # whichever backend app.py uses


def main():
    genuine, strangers = [], []

    print("=== GENUINE recordings (audio/<member_id>/) - used in training, so optimistic ===")
    for folder in sorted(p for p in AUDIO_DIR.iterdir() if p.is_dir() and p.name.lower() not in IGNORED):
        for wav in sorted(folder.glob("*.wav")):
            label, conf = score(wav)
            ok = app.resolve_member(label) == app.resolve_member(folder.name)
            genuine.append(conf if ok else 0.0)
            print(f"  {folder.name}/{wav.name:<28} -> {label}  {conf:.2f}  {'OK' if ok else 'WRONG PERSON'}")

    new_clips = sorted(SAMPLES_DIR.glob("new_member_*.wav")) if SAMPLES_DIR.exists() else []
    if new_clips:
        print("\n=== NEW genuine recordings (samples/new_member_XX_*.wav) - fair test ===")
        for wav in new_clips:
            expected = "_".join(wav.stem.split("_")[1:3])
            label, conf = score(wav)
            ok = app.resolve_member(label) == app.resolve_member(expected)
            genuine.append(conf if ok else 0.0)
            print(f"  {wav.name:<36} -> {label}  {conf:.2f}  {'OK' if ok else 'WRONG PERSON'}")

    stranger_files = sorted((AUDIO_DIR / "unknown").glob("*.wav")) if (AUDIO_DIR / "unknown").exists() else []
    if stranger_files:
        print("\n=== STRANGER recordings (audio/unknown/) - should be rejected ===")
        for wav in stranger_files:
            label, conf = score(wav)
            accepted = app.resolve_member(label) is not None
            strangers.append(conf if accepted else 0.0)
            print(f"  {wav.name:<36} -> best guess {label}  {conf:.2f}  "
                  f"{'!! would be accepted as a member' if accepted else 'rejected as unknown'}")

    print("\n=== SUMMARY ===")
    if genuine:
        print(f"  lowest genuine confidence : {min(genuine):.2f}  (a wrong person counts as 0.00)")
    if strangers:
        print(f"  highest stranger confidence accepted as a member: {max(strangers):.2f}")
    if genuine and strangers:
        lo, hi = min(genuine), max(strangers)
        if lo > hi:
            print(f"  -> clean gap. Suggested VOICE_THRESHOLD = {(lo + hi) / 2:.2f}")
        else:
            print("  -> genuine and stranger scores OVERLAP: no threshold separates them perfectly.")
            print(f"     To reject your stranger clip(s) you would need about {hi + 0.01:.2f}, "
                  "but some genuine clips would then be denied.")
    elif not strangers:
        print("  No stranger clip yet: put one in audio/unknown/ (e.g. audio/unknown/unknown_voice.wav).")


if __name__ == "__main__":
    main()