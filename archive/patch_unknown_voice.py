"""
patch_unknown_voice.py -- one-time patch for app.py so audio/unknown/ is used for STRANGER voices.

Run from the repo root:   python patch_unknown_voice.py
Backs up app.py first (app_backup_unknown.py). Safe to run twice.
"""
import re
import shutil
import sys
from pathlib import Path

p = Path("app.py")
if not p.exists():
    sys.exit("app.py not found - run this from the repo root.")
s = p.read_text(encoding="utf-8")
if "def _selftest_voice" not in s:
    sys.exit("Run patch_app_auto.py first (app.py has no auto voice selection yet).")
if not Path("app_backup_unknown.py").exists():
    shutil.copy(p, "app_backup_unknown.py")

NEW = '''def _stranger_clips():
    """Stranger recordings: audio/unknown/*.wav (+ samples/unknown_voice.wav if present)."""
    clips = sorted((AUDIO_DIR / "unknown").glob("*.wav")) if (AUDIO_DIR / "unknown").exists() else []
    old = ROOT / "samples" / "unknown_voice.wav"
    return clips + ([old] if old.exists() else [])


def _selftest_voice(backend):
    """
    Check a voice backend against the recordings in audio/<member_id>/ (audio/unknown is skipped)
    and against the stranger clips. Returns dict: accepted, total, stranger_ok (True/False/None), error.
    """
    res = {"accepted": 0, "total": 0, "stranger_ok": None, "error": ""}
    if AUDIO_DIR.exists():
        for folder in sorted(d for d in AUDIO_DIR.iterdir() if d.is_dir() and d.name.lower() != "unknown"):
            for wav in sorted(folder.glob("*.wav")):
                res["total"] += 1
                try:
                    label, conf = _run_voice_backend(backend, wav)
                    if resolve_member(label) == resolve_member(folder.name) and conf >= VOICE_THRESHOLD:
                        res["accepted"] += 1
                except Exception as e:
                    res["error"] = res["error"] or f"{type(e).__name__}: {e}"
    for clip in _stranger_clips():
        try:
            label, conf = _run_voice_backend(backend, clip)
            rejected = resolve_member(label) is None or conf < VOICE_THRESHOLD
            res["stranger_ok"] = rejected if res["stranger_ok"] is None else (res["stranger_ok"] and rejected)
        except Exception as e:
            res["error"] = res["error"] or f"{type(e).__name__}: {e}"
    return res


'''
if "def _stranger_clips" not in s:
    pat = re.compile(r'def _selftest_voice\(backend\):.*?(?=def _choose_voice_backend)', re.S)
    s = pat.sub(lambda m: NEW, s, count=1)
    s = s.replace('"no stranger clip yet"', '"no stranger clip yet (put one in audio/unknown/)"')
s = s.replace('"voice": "samples/unknown_voice.wav"', '"voice": "audio/unknown/unknown_voice.wav"')
p.write_text(s, encoding="utf-8")
print("app.py updated: stranger voices now come from audio/unknown/")
print("Scenario 4 uses audio/unknown/unknown_voice.wav - name your stranger clip exactly that.")