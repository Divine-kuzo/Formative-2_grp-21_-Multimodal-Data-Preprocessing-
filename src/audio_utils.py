import numpy as np
import librosa
import joblib
import pandas as pd

from pathlib import Path
from functools import lru_cache


# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

SAMPLE_RATE = 22050


def extract_audio_features(audio_path):
    """
    Extract the same 15 audio features used during model training.
    """

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    rolloff_mean = np.mean(rolloff)

    # RMS energy
    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)

    # Combine features
    features = {}

    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}"] = float(value)

    features["spectral_rolloff"] = float(rolloff_mean)
    features["rms_energy"] = float(rms_mean)

    return features


@lru_cache(maxsize=1)
def load_voice_artifacts():
    """
    Load the trained voice model and supporting artifacts.
    """

    model = joblib.load(
        MODEL_DIR / "voice_model.pkl"
    )

    encoder = joblib.load(
        MODEL_DIR / "voice_label_encoder.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "voice_feature_columns.pkl"
    )

    return model, encoder, feature_columns


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

    model, encoder, feature_columns = load_voice_artifacts()

    # Extract features
    features = extract_audio_features(audio_path)

    # Arrange features in the exact training order
    X = pd.DataFrame([features])
    X = X[feature_columns]

    # Predict speaker
    prediction = model.predict(X)[0]

    predicted_member = encoder.inverse_transform(
        [prediction]
    )[0]

    # Compare predicted and expected identities
    verified = str(predicted_member) == str(expected_member_id)

    return {
        "verified": bool(verified),
        "predicted_member_id": str(predicted_member),
        "expected_member_id": str(expected_member_id)
    }
