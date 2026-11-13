"""
check_face.py -- runs Member 2's face model on every image and prints what comes back.

Run from the repo root:   python check_face.py
  * images/member_XX/*.jpg   -> should be recognised as that member   (these were used in training, so optimistic)
  * images/unknown/*         -> should come out as "unknown" / low confidence
Use the printed confidences to choose FACE_THRESHOLD in app.py.
"""
from pathlib import Path

import app

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
EXT = {".jpg", ".jpeg", ".png"}

app.FACE_BACKEND = "real"
genuine, strangers, wrong = [], [], 0

print("RAW return of predict_face for the first image (to confirm the format):")
try:
    from src.image_utils import predict_face
    first = next(p for p in sorted(IMG.rglob("*")) if p.suffix.lower() in EXT)
    print(f"  {first.name} -> {predict_face(str(first))!r}\n")
except Exception as e:
    print(f"  could not run predict_face: {e}\n")

for folder in sorted(p for p in IMG.iterdir() if p.is_dir()):
    for img in sorted(f for f in folder.iterdir() if f.suffix.lower() in EXT):
        try:
            label, conf = app.predict_face_backend(img)
        except Exception as e:
            print(f"  {folder.name}/{img.name:<28} ERROR: {e}")
            continue
        key = app.resolve_member(label)
        if folder.name == "unknown":
            strangers.append(conf if key else 0.0)
            tag = "REJECTED-as-unknown" if key is None else "!! ACCEPTED AS A MEMBER"
        else:
            ok = key == folder.name
            wrong += not ok
            genuine.append(conf if ok else 0.0)
            tag = "OK" if ok else "WRONG PERSON"
        print(f"  {folder.name}/{img.name:<28} -> {str(label):<10} {conf:.2f}  {tag}")

print("\n=== SUMMARY ===")
if genuine:
    print(f"  members: {len(genuine) - wrong}/{len(genuine)} recognised correctly, lowest confidence {min(genuine):.2f}")
if strangers:
    print(f"  strangers: highest confidence accepted as a member = {max(strangers):.2f}")
    if max(strangers) == 0.0:
        print("  -> all stranger images were rejected by the face model itself.")