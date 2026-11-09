"""
app.py  --  Command-line simulation of the multimodal authentication flow

    face image  ->  voice sample  ->  product recommendation

Flow
----
1. User gives a face image      -> face model must recognise a known member
2. User gives a voice sample    -> voice model must match the SAME member
3. Both pass                    -> product model predicts the product for that member's customer profile
   Anything fails               -> ACCESS DENIED (the product model is never called)

Run (from the repo root)
------------------------
    python app.py --demo                           # runs all scenarios automatically
    python app.py                                  # interactive: asks for the files
    python app.py --face PATH --voice PATH        # one transaction

Switching from placeholders to the real models (the only things to edit)
-----------------------------------------------------------------------
  * FACE_BACKEND  : "placeholder"  ->  "real"      (when Member 2's src/face_model.py is in the repo)
  * VOICE_BACKEND : "placeholder"  ->  "member3"   (Member 3's src/audio_utils.py + models/voice_model.pkl)  <- use this
                                   ->  "fallback"  (my interim src/voice_fallback.py, only if Member 3's model is unavailable)
  * MEMBERS       : fill in real names, and the labels each model returns
  * DEMO_SCENARIOS: point the file paths at real images / audio
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so "from src.xxx import ..." works from anywhere

# CONFIG  (edit this section)
FACE_BACKEND = "placeholder"    # "placeholder" | "real"
VOICE_BACKEND = "fallback"   # "placeholder" | "fallback" | "member3"

FACE_THRESHOLD = 0.60           # minimum confidence to accept a face
VOICE_THRESHOLD = 0.60          # minimum confidence to accept a voice

# One entry per team member.
#   key          = the member_id used in audio_features.csv
#   name         = the member's real name (FILL THESE IN)
#   aliases      = any other labels the face/voice models may return for this person
#                  (check Member 2's label encoder: e.g. "racheal", "member_01", "1")
#   customer_id  = the customer profile from product_model_data.csv this member "acts as"
#                  (these four have one consistent product label; run
#                   `python src/product_model.py --list-customers` for others)
MEMBERS = {
    "member_01": {"name": "Member 1", "aliases": [], "customer_id": "A104"},
    "member_02": {"name": "Member 2", "aliases": [], "customer_id": "A105"},
    "member_03": {"name": "Member 3", "aliases": [], "customer_id": "A122"},
    "member_04": {"name": "Member 4", "aliases": [], "customer_id": "A125"},
}

# Files used by --demo. With the placeholder backends only the FILE NAMES matter
# (the files don't have to exist). With real backends, point these at real files.
DEMO_SCENARIOS = [
    {"title": "1. Authorised user - full transaction",
     "face": "samples/member_01_neutral.jpg", "voice": "audio/member_01/confirm_transaction.wav"},
    {"title": "2. Unauthorised face (stranger)",
     "face": "samples/unknown_face.jpg", "voice": "audio/member_01/confirm_transaction.wav"},
    {"title": "3. Authorised face, but someone else's voice",
     "face": "samples/member_02_smiling.jpg", "voice": "audio/member_03/confirm_transaction.wav"},
    {"title": "4. Authorised face, unauthorised voice (stranger)",
     "face": "samples/member_04_surprised.jpg", "voice": "samples/unknown_voice.wav"},
]

LOG_PATH = ROOT / "outputs" / "simulation_log.csv"


# Helpers: matching model labels to team members
def resolve_member(label):
    """Map whatever label a model returns (name, 'member_01', alias) to a MEMBERS key, or None."""
    if label is None:
        return None
    text = str(label).strip().lower()
    for key, info in MEMBERS.items():
        names = [key, info["name"], *info["aliases"]]
        if text in [n.lower() for n in names]:
            return key
    return None


def display_name(key):
    return MEMBERS[key]["name"] if key in MEMBERS else "unknown"


# ==========================================================================
# FACE backend
# ==========================================================================
def _placeholder_face(image_path):
    """
    PLACEHOLDER - does NOT look at the image. It guesses the person from the FILE NAME:
    a name containing 'member_02' (or a member's name/alias) -> that member;
    anything else (e.g. 'unknown_face.jpg') -> unknown.
    """
    stem = Path(image_path).name.lower()
    for key, info in MEMBERS.items():
        tokens = [key, info["name"].lower().replace(" ", "_"), *[a.lower() for a in info["aliases"]]]
        if any(t in stem for t in tokens):
            return key, 0.95
    return "unknown", 0.20


def predict_face_backend(image_path):
    """Returns (label, confidence)."""
    if FACE_BACKEND == "real":
        # Member 2's file must provide:  predict_face(image_path) -> (label, confidence)
        from src.face_model import predict_face
        return predict_face(str(image_path))
    return _placeholder_face(image_path)


# VOICE backend
def _placeholder_voice(audio_path):
    """PLACEHOLDER - does NOT listen to the audio; guesses the person from the FILE NAME."""
    stem = Path(audio_path).name.lower()
    for key, info in MEMBERS.items():
        tokens = [key, info["name"].lower().replace(" ", "_"), *[a.lower() for a in info["aliases"]]]
        if any(t in stem for t in tokens):
            return key, 0.95
    return "unknown", 0.20


def _split_voice_artifacts(artifacts):
    """
    Member 3's load_voice_artifacts() has changed shape before (3 values, then 2).
    Recognise each item by what it is instead of by position:
      model   = has predict() (a sklearn model or Pipeline)
      encoder = has inverse_transform but no predict (a LabelEncoder), optional
      columns = a list of feature names (falls back to model.feature_names_in_)
    Returns (model, encoder_or_None, columns).
    """
    items = list(artifacts.values()) if isinstance(artifacts, dict) else (
        list(artifacts) if isinstance(artifacts, (tuple, list)) else [artifacts])
    model = encoder = columns = None
    for it in items:
        if isinstance(it, (list, tuple)):
            columns = list(it)
        elif hasattr(it, "predict"):
            model = it
        elif hasattr(it, "inverse_transform"):
            encoder = it
        elif hasattr(it, "tolist"):
            columns = list(it)
    if model is None:
        raise RuntimeError("load_voice_artifacts() did not return a model with predict().")
    if columns is None:
        columns = list(getattr(model, "feature_names_in_", []))
        if not columns:
            raise RuntimeError("Could not find the voice feature column names.")
    return model, encoder, columns


def _confidences(model, X):
    """
    Return (classes, scores) where scores sum to 1.
    Uses predict_proba when the model has it. Models without it (e.g. an SVM trained
    without probability=True) fall back to a softmax of decision_function, which is a
    RELATIVE score, not a calibrated probability. Last resort: predict() with 1.0.
    """
    import numpy as np
    classes = getattr(model, "classes_", None)
    if hasattr(model, "predict_proba"):
        return classes, model.predict_proba(X)[0]
    if hasattr(model, "decision_function"):
        d = np.atleast_1d(model.decision_function(X)[0]).astype(float)
        if d.size == 1:                       # binary case
            d = np.array([-d[0], d[0]])
        e = np.exp(d - d.max())
        return classes, e / e.sum()
    pred = model.predict(X)[0]
    return [pred], np.array([1.0])


def _member3_voice(audio_path):
    """
    Adapter for Member 3's code (src/audio_utils.py + models/voice_model.pkl).
    Their verify_voice() only returns verified True/False with no confidence, so this
    reuses THEIR feature extraction and THEIR trained model and works out a confidence
    score itself. None of Member 3's files are modified.
    """
    import numpy as np
    from src.audio_utils import extract_audio_features, load_voice_artifacts

    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model, encoder, columns = _split_voice_artifacts(load_voice_artifacts())
    feats = extract_audio_features(audio_path)
    X = np.asarray([[feats[c] for c in columns]], dtype=float)   # same as Member 3's verify_voice
    classes, scores = _confidences(model, X)
    # Use the SAME answer as Member 3's verify_voice (model.predict) and report that
    # class's score as the confidence. (With very little data, an SVM's predict() and
    # predict_proba() can disagree, so we don't take the argmax of the scores.)
    cls = model.predict(X)[0]
    if classes is not None and cls in list(classes):
        i = list(classes).index(cls)
    else:
        i = int(np.argmax(scores))
    label = encoder.inverse_transform([cls])[0] if encoder is not None else cls
    return str(label), float(scores[i])


def predict_voice_backend(audio_path):
    """Returns (label, confidence)."""
    if VOICE_BACKEND == "fallback":
        from src.voice_fallback import predict_voice
        return predict_voice(str(audio_path))
    if VOICE_BACKEND == "member3":
        return _member3_voice(audio_path)
    return _placeholder_voice(audio_path)


def _require_file(path, kind):
    """Real backends need the file to exist; placeholders only need a name."""
    using_real = (FACE_BACKEND == "real") if kind == "face" else (VOICE_BACKEND != "placeholder")
    if using_real and not Path(path).exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")


# The transaction
def line(text=""):
    print(text)


def run_transaction(face_path, voice_path):
    """
    Run one attempt. Returns a dict: {"outcome": "APPROVED" | "DENIED_FACE" | "DENIED_VOICE" | "ERROR", ...}
    """
    result = {"face_file": str(face_path), "voice_file": str(voice_path),
              "face_label": "", "face_conf": "", "voice_label": "", "voice_conf": "",
              "customer_id": "", "product": "", "outcome": ""}
    line("-" * 60)

    # ---- Step 1: face -------------------------------------------------------
    line("[Step 1/3] Face verification ...")
    try:
        _require_file(face_path, "face")
        f_label, f_conf = predict_face_backend(face_path)
    except Exception as e:
        line(f"   ERROR: {e}")
        result["outcome"] = "ERROR"
        return result
    f_key = resolve_member(f_label)
    result.update(face_label=str(f_label), face_conf=f"{f_conf:.2f}")
    if f_key is None or f_conf < FACE_THRESHOLD:
        line(f"   Face not recognised (best guess: {f_label}, confidence {f_conf:.0%}).")
        line("   >>> ACCESS DENIED - unknown face. Product model not accessed.")
        result["outcome"] = "DENIED_FACE"
        return result
    line(f"   Face recognised: {display_name(f_key)} (confidence {f_conf:.0%})")

    # ---- Step 2: voice ------------------------------------------------------
    line("[Step 2/3] Voice verification (phrase: 'Yes, approve' / 'Confirm transaction') ...")
    try:
        _require_file(voice_path, "voice")
        v_label, v_conf = predict_voice_backend(voice_path)
    except Exception as e:
        line(f"   ERROR: {e}")
        result["outcome"] = "ERROR"
        return result
    v_key = resolve_member(v_label)
    result.update(voice_label=str(v_label), voice_conf=f"{v_conf:.2f}")
    if v_key is None or v_conf < VOICE_THRESHOLD:
        line(f"   Voice not recognised (best guess: {v_label}, confidence {v_conf:.0%}).")
        line("   >>> ACCESS DENIED - unauthorised voice. Prediction not allowed.")
        result["outcome"] = "DENIED_VOICE"
        return result
    if v_key != f_key:
        line(f"   Voice belongs to {display_name(v_key)}, but the face was {display_name(f_key)}.")
        line("   >>> ACCESS DENIED - face and voice do not match. Prediction not allowed.")
        result["outcome"] = "DENIED_VOICE"
        return result
    line(f"   Voice approved: {display_name(v_key)} (confidence {v_conf:.0%})")

    # ---- Step 3: product prediction ----------------------------------------
    line("[Step 3/3] Both checks passed - running product recommendation ...")
    try:
        from src.product_model import predict_product
        customer_id = MEMBERS[f_key]["customer_id"]
        pred = predict_product(customer_id)
    except Exception as e:
        line(f"   ERROR: {e}")
        result["outcome"] = "ERROR"
        return result
    line(f"   >>> ACCESS GRANTED for {display_name(f_key)} (customer {customer_id})")
    line(f"   Recommended product: {pred['product']}  (confidence {pred['confidence']:.0%})")
    line("   Other likely products: " + ", ".join(f"{p} {c:.0%}" for p, c in pred["top3"][1:]))
    result.update(customer_id=customer_id, product=pred["product"], outcome="APPROVED")
    return result


def log_result(res):
    """Append every attempt to outputs/simulation_log.csv (handy evidence for the report/video)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", *res.keys()])
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), **res})


# Modes
def banner():
    line("=" * 60)
    line(" USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")
    line(f" face backend: {FACE_BACKEND} | voice backend: {VOICE_BACKEND}")
    line("=" * 60)


def run_demo():
    summary = []
    for sc in DEMO_SCENARIOS:
        line(f"\n##### {sc['title']}")
        line(f"      face : {sc['face']}")
        line(f"      voice: {sc['voice']}")
        res = run_transaction(sc["face"], sc["voice"])
        log_result(res)
        summary.append((sc["title"], res["outcome"]))
    line("\n" + "=" * 60)
    line(" DEMO SUMMARY")
    for title, outcome in summary:
        line(f"  {outcome:<13} {title}")
    line("=" * 60)


def run_interactive():
    while True:
        face = input("\nPath to face image (or 'q' to quit): ").strip().strip('"')
        if face.lower() == "q":
            break
        voice = input("Path to voice sample (.wav):        ").strip().strip('"')
        res = run_transaction(face, voice)
        log_result(res)
        if input("\nTry another transaction? (y/n): ").strip().lower() != "y":
            break
    line("Goodbye.")


def main():
    ap = argparse.ArgumentParser(description="Multimodal authentication + product recommendation")
    ap.add_argument("--demo", action="store_true", help="run all demo scenarios")
    ap.add_argument("--face", help="path to a face image")
    ap.add_argument("--voice", help="path to a voice sample")
    args = ap.parse_args()

    banner()
    if args.demo:
        run_demo()
    elif args.face and args.voice:
        log_result(run_transaction(args.face, args.voice))
    elif args.face or args.voice:
        sys.exit("Give both --face and --voice, or neither for interactive mode.")
    else:
        run_interactive()


if __name__ == "__main__":
    main()