"""
patch_app_auto.py -- one-time patch: makes the voice step choose its own model ("auto").

Run from the repo root:   python patch_app_auto.py
Keeps everything you already set (MEMBERS, customer IDs, thresholds). Makes a backup first
(app_backup_auto.py). Safe to run twice.

What "auto" does when the app starts:
  1. Tries Member 3's model (src/audio_utils.py + models/voice_model.pkl) and my model
     (src/voice_fallback.py) on every recording in audio/<member_id>/.
  2. If samples/unknown_voice.wav exists, it also checks that the stranger is rejected.
  3. Uses the first one that passes (Member 3's model is preferred) and prints which one and why.
So whatever Member 3 pushes (new CSV, new model, new audio_utils.py), the app picks a voice model
that actually works on your recordings and tells you if neither does.
"""
import re
import shutil
import sys
from pathlib import Path

p = Path("app.py")
if not p.exists():
    sys.exit("app.py not found - run this from the repo root.")
s = p.read_text(encoding="utf-8")
if not Path("app_backup_auto.py").exists():
    shutil.copy(p, "app_backup_auto.py")

BLOCK = '''AUDIO_DIR = ROOT / "audio"
STRANGER_VOICE = ROOT / "samples" / "unknown_voice.wav"
MIN_GENUINE_SHARE = 0.85        # share of the recordings in audio/ a voice model must accept
_VOICE_CHOICE = None


def _run_voice_backend(backend, audio_path):
    """Run ONE named voice backend. Returns (label, confidence)."""
    if backend == "fallback":
        from src.voice_fallback import predict_voice
        return predict_voice(str(audio_path))
    if backend == "member3":
        return _member3_voice(audio_path)
    return _placeholder_voice(audio_path)


def _selftest_voice(backend):
    """
    Check a voice backend against the recordings in audio/<member_id>/ (and the stranger clip).
    Returns dict: accepted, total, stranger_ok (True/False/None), error (first error or "").
    """
    res = {"accepted": 0, "total": 0, "stranger_ok": None, "error": ""}
    if AUDIO_DIR.exists():
        for folder in sorted(d for d in AUDIO_DIR.iterdir() if d.is_dir()):
            for wav in sorted(folder.glob("*.wav")):
                res["total"] += 1
                try:
                    label, conf = _run_voice_backend(backend, wav)
                    if resolve_member(label) == resolve_member(folder.name) and conf >= VOICE_THRESHOLD:
                        res["accepted"] += 1
                except Exception as e:
                    res["error"] = res["error"] or f"{type(e).__name__}: {e}"
    if STRANGER_VOICE.exists():
        try:
            label, conf = _run_voice_backend(backend, STRANGER_VOICE)
            res["stranger_ok"] = resolve_member(label) is None or conf < VOICE_THRESHOLD
        except Exception as e:
            res["error"] = res["error"] or f"{type(e).__name__}: {e}"
    return res


def _choose_voice_backend():
    """VOICE_BACKEND = "auto": pick the first backend that passes the self-test. Cached."""
    global _VOICE_CHOICE
    if _VOICE_CHOICE is not None:
        return _VOICE_CHOICE
    results = {}
    for backend in ("member3", "fallback"):
        results[backend] = _selftest_voice(backend)
    def passes(r):
        return (r["total"] > 0 and r["accepted"] >= MIN_GENUINE_SHARE * r["total"]
                and r["stranger_ok"] is not False)
    chosen = next((b for b in ("member3", "fallback") if passes(results[b])), None)
    line("[voice] self-test on the recordings in audio/:")
    for b, r in results.items():
        stranger = {True: "stranger rejected", False: "STRANGER ACCEPTED", None: "no stranger clip yet"}[r["stranger_ok"]]
        extra = f"  ({r['error']})" if r["error"] and not r["accepted"] else ""
        line(f"   {b:<9} {r['accepted']}/{r['total']} accepted, {stranger}{extra}")
    if chosen is None:
        chosen = max(results, key=lambda b: (results[b]["stranger_ok"] is not False, results[b]["accepted"]))
        line(f"[voice] WARNING: no voice model passed. Using '{chosen}' anyway. "
             "Train mine with: python src/voice_fallback.py --from-audio")
    else:
        line(f"[voice] using: {chosen}")
    _VOICE_CHOICE = chosen
    return chosen


def predict_voice_backend(audio_path):
    """Returns (label, confidence)."""
    backend = VOICE_BACKEND
    if backend == "auto":
        backend = _choose_voice_backend()
    return _run_voice_backend(backend, audio_path)
'''

if "_choose_voice_backend" not in s:
    pat = re.compile(r'def predict_voice_backend\(audio_path\):.*?return _placeholder_voice\(audio_path\)\n', re.S)
    if not pat.search(s):
        sys.exit("Could not find predict_voice_backend in app.py - send me your app.py.")
    s = pat.sub(lambda m: BLOCK, s, count=1)
    print("auto voice selection added")
else:
    print("auto voice selection already present")

old = '    line(f" face backend: {FACE_BACKEND} | voice backend: {VOICE_BACKEND}")\n    line("=" * 60)\n'
new = old + '    if VOICE_BACKEND == "auto":\n        _choose_voice_backend()\n'
if old in s and "_choose_voice_backend()\n" not in s.split("def banner")[1].split("def ")[0]:
    s = s.replace(old, new, 1)
    print("banner updated")

s, n = re.subn(r'^VOICE_BACKEND\s*=.*$',
               'VOICE_BACKEND = "auto"          # "auto" | "member3" | "fallback" | "placeholder"',
               s, count=1, flags=re.M)
print('VOICE_BACKEND set to "auto"' if n else "WARNING: VOICE_BACKEND line not found")

p.write_text(s, encoding="utf-8")
print("done. Backup saved as app_backup_auto.py")