"""
voice_fallback.py  --  INTERIM voiceprint verification model

Purpose
-------
Lets the command-line app run end to end while Member 3's own voice model
is not ready. It trains on data/audio_features.csv (15 features per
recording: 13 MFCC means, spectral roll-off, RMS energy).

When Member 3 shares their model, set VOICE_BACKEND = "member3" in app.py
and this file is no longer used (keep it in the repo as a fallback, or delete it).

Run (from the repo root):
    python src/voice_fallback.py            # train, evaluate, save
    python src/voice_fallback.py --check    # re-predict the recordings in audio/ (sanity test)

Contract used by app.py:
    predict_voice(audio_path) -> (member_id, confidence)

IMPORTANT: extract_features() below must compute features EXACTLY like
Member 3 did when making audio_features.csv (same sample rate, n_mfcc=13,
mean over time). Ask Member 3 for their extraction code and copy it in if
the settings differ -- otherwise predictions on new recordings will be wrong.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_fallback.pkl"
AUDIO_DIR = ROOT / "audio"

FEATURE_COLS = [f"mfcc_{i}" for i in range(1, 14)] + ["spectral_rolloff", "rms_energy"]
LABEL_COL = "member_id"
SAMPLE_RATE = 22050  # librosa default; change if Member 3 used something else


# --------------------------------------------------------------------------
# Feature extraction from a .wav file (needs: pip install librosa soundfile)
# --------------------------------------------------------------------------
def extract_features(audio_path):
    """Return a 1-row DataFrame with the same 15 columns as audio_features.csv."""
    import librosa  # imported here so training from the CSV works without librosa

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    values = list(mfcc) + [rolloff, rms]
    return pd.DataFrame([values], columns=FEATURE_COLS)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train():
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    n_classes = len(le.classes_)
    print(f"[data] {len(df)} recordings, {n_classes} members: {list(le.classes_)}")
    if len(df) < 20:
        print("[warn] very few recordings -> results are only a rough indication. "
              "Ask Member 3 for the augmented rows (pitch shift, noise, ...).")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    # Leave-one-out: train on all recordings except one, test on that one.
    rows = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", clf)])
        probas = np.zeros((len(X), n_classes))
        for tr, te in LeaveOneOut().split(X):
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[0]
            full = np.zeros(n_classes)
            full[pipe.named_steps["model"].classes_] = p
            probas[te[0]] = full
        pred = probas.argmax(axis=1)
        rows[name] = {
            "loo_accuracy": accuracy_score(y, pred),
            "loo_f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
            "loo_log_loss": log_loss(y, np.clip(probas, 1e-6, 1), labels=list(range(n_classes))),
        }
    results = pd.DataFrame(rows).T.round(4)
    print("\n=== Leave-one-out evaluation ===")
    print(results.to_string())

    best = results.sort_values(["loo_accuracy", "loo_log_loss"], ascending=[False, True]).index[0]
    print(f"\n[best] {best}")

    final = Pipeline([("scale", StandardScaler()), ("model", candidates[best])])
    final.fit(X, y)

    # Stranger rejection ("open-set" check). A normal classifier ALWAYS picks one of the
    # known members, even for noise or a stranger. So we also store the training
    # recordings and, at prediction time, reject any sample that is too far from every
    # recording of the member it was matched to. TAU = 1.25 x the largest distance
    # seen between two recordings of the SAME member.
    Z = final.named_steps["scale"].transform(X)
    same = [np.linalg.norm(Z[i] - Z[j]) for i in range(len(Z)) for j in range(i + 1, len(Z))
            if y[i] == y[j]]
    tau = 1.25 * max(same) if same else np.inf
    print(f"[gate] distance threshold TAU = {tau:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "label_encoder": le, "model_name": best,
                 "metrics": results.loc[best].to_dict(),
                 "train_Z": Z, "train_y": y, "tau": float(tau)}, MODEL_PATH)
    (ROOT / "outputs").mkdir(exist_ok=True)
    results.to_csv(ROOT / "outputs" / "voice_fallback_metrics.csv")
    print(f"[saved] {MODEL_PATH}")
    return results


# --------------------------------------------------------------------------
# Prediction (used by app.py)
# --------------------------------------------------------------------------
_ART = None


def _load():
    global _ART
    if _ART is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python src/voice_fallback.py")
        _ART = joblib.load(MODEL_PATH)
    return _ART


def predict_voice_from_features(features_df):
    """Return (member_id, confidence), or ("unknown", 0.0) if the sample is too far from the matched member."""
    art = _load()
    X = features_df[FEATURE_COLS]
    proba = art["pipeline"].predict_proba(X)[0]
    i = int(np.argmax(proba))
    # distance (in standardised feature space) to the closest recording of that member
    z = art["pipeline"].named_steps["scale"].transform(X)[0]
    dists = np.linalg.norm(art["train_Z"][art["train_y"] == i] - z, axis=1)
    if dists.min() > art["tau"]:
        return "unknown", 0.0
    return art["label_encoder"].classes_[i], float(proba[i])


def predict_voice(audio_path):
    """Return (member_id, confidence) for a .wav file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return predict_voice_from_features(extract_features(audio_path))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="predict every recording in audio_features.csv "
                         "(needs the .wav files in audio/<member_id>/ or audio/<member_id>_<filename>)")
    args = ap.parse_args()

    if args.check:
        df = pd.read_csv(CSV_PATH)
        correct = 0
        for _, r in df.iterrows():
            # the same filename (e.g. yes_approve.wav) exists for every member,
            # so look for it inside a member folder first
            options = [AUDIO_DIR / r[LABEL_COL] / r["filename"],
                       AUDIO_DIR / f"{r[LABEL_COL]}_{r['filename']}",
                       AUDIO_DIR / r["filename"]]
            f = next((o for o in options if o.exists()), None)
            if f is None:
                print(f"missing: {options[0]}")
                continue
            who, conf = predict_voice(f)
            correct += who == r[LABEL_COL]
            print(f"{r['filename']:<28} true={r[LABEL_COL]}  pred={who}  conf={conf:.2f}")
        print(f"{correct}/{len(df)} correct")
    else:
        train()