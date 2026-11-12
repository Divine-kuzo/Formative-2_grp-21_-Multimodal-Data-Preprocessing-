"""
voice_fallback.py  --  voiceprint verification model used by app.py (VOICE_BACKEND = "fallback")

Two ways to train it (run from the repo root):

  python src/voice_fallback.py --from-audio   RECOMMENDED. Reads every .wav in audio/<member_id>/,
                                              creates augmented copies (pitch up/down, slower/faster,
                                              background noise) and trains on those. Features are
                                              computed by the SAME code that app.py uses on live
                                              recordings, so training and prediction always match.
                                              Also saves data/voice_training_features.csv.
  python src/voice_fallback.py                trains from data/audio_features.csv (Member 3's file).

  python src/voice_fallback.py --check        re-predicts every recording in audio/ (sanity test)

Contract used by app.py:   predict_voice(audio_path) -> (member_id, confidence)
Needs: pip install librosa soundfile
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
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "audio_features.csv"
TRAIN_CSV_OUT = ROOT / "data" / "voice_training_features.csv"
MODEL_PATH = ROOT / "models" / "voice_fallback.pkl"
AUDIO_DIR = ROOT / "audio"

FEATURE_COLS = [f"mfcc_{i}" for i in range(1, 14)] + ["spectral_rolloff", "rms_energy"]
LABEL_COL = "member_id"
SAMPLE_RATE = 22050   # every recording is resampled to this rate (training AND prediction)
IGNORED_FOLDERS = {"unknown"}   # audio/unknown/ holds STRANGER clips - never trained as a member


def member_folders():
    """audio/<member_id>/ folders (the stranger folder audio/unknown is skipped)."""
    if not AUDIO_DIR.exists():
        return []
    return sorted(p for p in AUDIO_DIR.iterdir() if p.is_dir() and p.name.lower() not in IGNORED_FOLDERS)


def stranger_clips():
    """Stranger recordings: audio/unknown/*.wav (plus samples/unknown_voice.wav if you have one)."""
    clips = sorted((AUDIO_DIR / "unknown").glob("*.wav")) if (AUDIO_DIR / "unknown").exists() else []
    extra = ROOT / "samples" / "unknown_voice.wav"
    return clips + ([extra] if extra.exists() else [])


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def features_from_signal(y, sr):
    """15 numbers: 13 MFCC means, spectral roll-off mean, RMS energy mean."""
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    return list(mfcc) + [float(rolloff), float(rms)]


def extract_features(audio_path):
    """1-row DataFrame with the 15 feature columns for a .wav file."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return pd.DataFrame([features_from_signal(y, sr)], columns=FEATURE_COLS)


def augmented_versions(y, sr, seed=0):
    """5 augmented copies of a recording: pitch +/-2 semitones, speed 0.9x / 1.1x, noise (20 dB SNR)."""
    import librosa
    rng = np.random.default_rng(seed)
    noise_power = np.mean(y ** 2) / (10 ** (20 / 10))
    return {
        "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "slower": librosa.effects.time_stretch(y, rate=0.9),
        "faster": librosa.effects.time_stretch(y, rate=1.1),
        "noise": y + rng.normal(0, np.sqrt(noise_power), len(y)),
    }


def build_table_from_audio(augment=True):
    """Features for every audio/<member_id>/*.wav (+ augmented copies)."""
    import librosa
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"{AUDIO_DIR} not found")
    rows = []
    for folder in member_folders():
        for wav in sorted(folder.glob("*.wav")):
            y, sr = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
            versions = {"original": y}
            if augment:
                versions.update(augmented_versions(y, sr))
            for aug, sig in versions.items():
                row = dict(zip(FEATURE_COLS, features_from_signal(sig, sr)))
                row.update({LABEL_COL: folder.name, "source": f"{folder.name}/{wav.name}",
                            "augmentation": aug})
                rows.append(row)
            print(f"  processed {folder.name}/{wav.name}")
    if not rows:
        raise RuntimeError("No .wav files found in audio/<member_id>/")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(from_audio=False):
    if from_audio:
        print("[data] building features from audio/ with augmentation ...")
        df = build_table_from_audio()
        TRAIN_CSV_OUT.parent.mkdir(exist_ok=True)
        df.to_csv(TRAIN_CSV_OUT, index=False)
        print(f"[saved] {TRAIN_CSV_OUT}")
    else:
        df = pd.read_csv(CSV_PATH)

    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    n = len(le.classes_)
    is_orig = (df["augmentation"] == "original").values if "augmentation" in df else np.ones(len(df), bool)
    print(f"[data] {len(df)} training rows ({int(is_orig.sum())} original recordings), "
          f"{n} members: {list(le.classes_)}")

    if from_audio:
        splitter = LeaveOneGroupOut().split(X, y, groups=df["source"])
        print("[eval] leave-one-RECORDING-out: each original recording AND all its augmented copies "
              "are held out together, then the model is tested on that recording.")
    else:
        splitter = LeaveOneOut().split(X)
        print("[eval] leave-one-row-out (augmented copies of the same recording can leak into training)")
    splits = list(splitter)

    candidates = {
        "Logistic Regression (C=1)": LogisticRegression(max_iter=5000, C=1.0),
        "Logistic Regression (C=10)": LogisticRegression(max_iter=5000, C=10.0),
        "Logistic Regression (C=100)": LogisticRegression(max_iter=5000, C=100.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    rows = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", clf)])
        probas = np.full((len(X), n), np.nan)
        for tr, te in splits:
            pipe.fit(X.iloc[tr], y[tr])
            p = np.zeros((len(te), n))
            p[:, pipe.named_steps["model"].classes_] = pipe.predict_proba(X.iloc[te])
            probas[te] = p
        m = is_orig & ~np.isnan(probas).any(axis=1)
        pred = probas[m].argmax(axis=1)
        full_fit = Pipeline([("scale", StandardScaler()), ("model", clf)]).fit(X, y)
        rows[name] = {
            "accuracy": accuracy_score(y[m], pred),
            "f1_weighted": f1_score(y[m], pred, average="weighted", zero_division=0),
            "log_loss": log_loss(y[m], np.clip(probas[m], 1e-6, 1), labels=list(range(n))),
            "fits_own_recordings": accuracy_score(y[is_orig], full_fit.predict(X[is_orig])),
        }
    results = pd.DataFrame(rows).T.round(4)
    print("\n=== Evaluation (accuracy/F1/loss = on held-out recordings; fits_own_recordings = "
          "accuracy on the recordings it was trained on) ===")
    print(results.to_string())

    # The demo needs the model to recognise the recordings it was trained on, so only models
    # that do (fits_own_recordings = 1.0) are eligible; among them pick the best held-out score.
    eligible = results[results["fits_own_recordings"] >= 0.999]
    if eligible.empty:
        print("[warn] no model recognises all its own recordings; picking the closest one")
        eligible = results[results["fits_own_recordings"] == results["fits_own_recordings"].max()]
    best = eligible.sort_values(["accuracy", "log_loss"], ascending=[False, True]).index[0]
    print(f"\n[best] {best}")
    final = Pipeline([("scale", StandardScaler()), ("model", candidates[best])])
    final.fit(X, y)

    # Stranger gate: a classifier always picks a member, even for a stranger or noise.
    # So we also store the ORIGINAL recordings and reject any sample that is further from
    # the matched member's recordings than 1.25 x the largest distance between two
    # recordings of the SAME member.
    Zall = final.named_steps["scale"].transform(X)
    gZ, gy = Zall[is_orig], y[is_orig]
    same = [np.linalg.norm(gZ[i] - gZ[j]) for i in range(len(gZ)) for j in range(i + 1, len(gZ))
            if gy[i] == gy[j]]
    tau = 1.25 * max(same) if same else np.inf
    print(f"[gate] distance threshold TAU = {tau:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "label_encoder": le, "model_name": best,
                 "metrics": results.loc[best].to_dict(),
                 "gate_Z": gZ, "gate_y": gy, "tau": float(tau)}, MODEL_PATH)
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
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python src/voice_fallback.py --from-audio")
        _ART = joblib.load(MODEL_PATH)
    return _ART


def predict_voice_from_features(features_df):
    """(member_id, confidence), or ("unknown", 0.0) if too far from the matched member's recordings."""
    art = _load()
    X = features_df[FEATURE_COLS]
    proba = art["pipeline"].predict_proba(X)[0]
    i = int(np.argmax(proba))
    z = art["pipeline"].named_steps["scale"].transform(X)[0]
    gZ = art.get("gate_Z", art.get("train_Z"))
    gy = art.get("gate_y", art.get("train_y"))
    if np.linalg.norm(gZ[gy == i] - z, axis=1).min() > art["tau"]:
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
    ap.add_argument("--from-audio", action="store_true",
                    help="train from the .wav files in audio/ (+ augmentation)  [recommended]")
    ap.add_argument("--check", action="store_true", help="re-predict every recording in audio/")
    args = ap.parse_args()

    if args.check:
        ok = tot = 0
        for folder in member_folders():
            for wav in sorted(folder.glob("*.wav")):
                who, conf = predict_voice(wav)
                ok += who == folder.name
                tot += 1
                print(f"{folder.name}/{wav.name:<28} -> {who}  {conf:.2f}")
        print(f"{ok}/{tot} correct")
    else:
        train(from_audio=args.from_audio)