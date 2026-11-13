"""
patch_app.py -- one-time patch for app.py (keeps your MEMBERS names, customer IDs and VOICE_BACKEND).

Run from the repo root:   python patch_app.py
It makes a backup copy (app_backup.py) first, then:
  1. adds the connection to Member 2's face model (src/image_utils.py -> predict_face)
  2. sets FACE_BACKEND = "real"
  3. points DEMO_SCENARIOS at the real images in images/
Running it twice is safe.
"""
import re
import shutil
import sys
from pathlib import Path

p = Path("app.py")
if not p.exists():
    sys.exit("app.py not found - run this from the repo root.")
s = p.read_text(encoding="utf-8")
if not Path("app_backup.py").exists():      # keep the FIRST backup only
    shutil.copy(p, "app_backup.py")

FACE_BLOCK = '''def _normalize_face_result(res):
    """
    Turn whatever Member 2's predict_face() returns into (label, confidence).
    Accepts: (label, confidence) | (label, confidence, ...) | a dict | just a label.
    """
    if isinstance(res, dict):
        low = {str(k).lower(): v for k, v in res.items()}
        label = next((low[k] for k in ("label", "member_id", "member", "name", "predicted",
                                       "prediction", "identity", "person") if k in low), None)
        conf = next((low[k] for k in ("confidence", "probability", "proba", "prob", "score")
                     if k in low), None)
    elif isinstance(res, (tuple, list)):
        label = next((x for x in res if isinstance(x, str)), None)
        nums = [x for x in res if isinstance(x, (int, float)) and not isinstance(x, bool)]
        conf = nums[0] if nums else None
    else:
        label, conf = res, None
    if label is None or str(label).strip().lower() in ("", "none", "unknown", "unauthorized",
                                                        "unauthorised", "non_member", "non-member"):
        return "unknown", float(conf) if conf is not None else 0.0
    return str(label), float(conf) if conf is not None else 1.0


def _member2_face(image_path):
    """Adapter for Member 2's code: src/image_utils.py -> predict_face(image_path)."""
    from src.image_utils import predict_face
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"face file not found: {image_path}")
    return _normalize_face_result(predict_face(str(image_path)))


def predict_face_backend(image_path):
    """Returns (label, confidence)."""
    if FACE_BACKEND == "real":
        return _member2_face(image_path)
    return _placeholder_face(image_path)
'''

if "_member2_face" not in s:
    pat = re.compile(r'def predict_face_backend\(image_path\):.*?return _placeholder_face\(image_path\)\n', re.S)
    if not pat.search(s):
        sys.exit("Could not find predict_face_backend in app.py - send me your app.py.")
    s = pat.sub(lambda m: FACE_BLOCK, s, count=1)
    print("face adapter added")
else:
    print("face adapter already present")

s, n = re.subn(r'^FACE_BACKEND\s*=\s*"[^"]*"', 'FACE_BACKEND = "real"', s, flags=re.M)
print("FACE_BACKEND set to real" if n else "WARNING: FACE_BACKEND line not found")

DEMO = '''DEMO_SCENARIOS = [
    {"title": "1. Authorised user - full transaction",
     "face": "images/member_01/member_01_neutral.jpg", "voice": "audio/member_01/confirm_transaction.wav"},
    {"title": "2. Unauthorised face (stranger)",
     "face": "images/unknown/non_member_1.png", "voice": "audio/member_01/confirm_transaction.wav"},
    {"title": "3. Authorised face, but someone else's voice",
     "face": "images/member_02/member_02_smile.jpg", "voice": "audio/member_03/confirm_transaction.wav"},
    {"title": "4. Authorised face, unauthorised voice (stranger)",
     "face": "images/member_04/member_04_surprised.jpg", "voice": "samples/unknown_voice.wav"},
]
'''
pat = re.compile(r'DEMO_SCENARIOS\s*=\s*\[.*?\n\]\n', re.S)
if pat.search(s):
    s = pat.sub(lambda m: DEMO, s, count=1)
    print("DEMO_SCENARIOS updated")
else:
    print("WARNING: DEMO_SCENARIOS block not found")

p.write_text(s, encoding="utf-8")
print("done. Backup saved as app_backup.py")