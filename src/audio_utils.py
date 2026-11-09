import numpy as np
import librosa
import joblib

from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


def extract_audio_features(audio_path):
    """
    Extract the same 15 audio features used during Colab training.
    Preserve the original recording sample rate.
    """
    audio, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    rolloff_mean = np.mean(rolloff)

    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    features = {}

    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}"] = float(value)

    features["spectral_rolloff"] = float(rolloff_mean)
    features["rms_energy"] = float(rms_mean)

    return features


@lru_cache(maxsize=1)
def load_voice_artifacts():
    """
    Load the new model and its feature column order.
    """
    model_path = MODEL_DIR / "voice_model.pkl"
    columns_path = MODEL_DIR / "voice_feature_columns.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not columns_path.is_file():
        raise FileNotFoundError(
            f"Feature columns not found: {columns_path}"
        )

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)

    return model, feature_columns


def verify_voice(audio_path, expected_member_id):
    """
    Predict the speaker and compare with the expected member.

    Returns a dictionary containing the verification result.
    """
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model, feature_columns = load_voice_artifacts()

    features = extract_audio_features(audio_path)

    X = np.asarray(
        [[features[column] for column in feature_columns]],
        dtype=float
    )

    prediction = model.predict(X)[0]

    predicted_member = str(prediction)

    verified = predicted_member == str(expected_member_id)

    return {
        "verified": bool(verified),
        "predicted_member_id": predicted_member,
        "expected_member_id": str(expected_member_id)
    }