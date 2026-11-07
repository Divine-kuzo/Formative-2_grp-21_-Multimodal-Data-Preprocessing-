
import joblib
import pandas as pd

from pathlib import Path
from .features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "voice_verification_model.pkl"
)

encoder = joblib.load(
    BASE_DIR / "voice_label_encoder.pkl"
)

feature_columns = joblib.load(
    BASE_DIR / "voice_feature_columns.pkl"
)

def verify_voice(audio_path, expected_member_id):

    features = extract_features(audio_path)

    X = pd.DataFrame([features])

    X = X[feature_columns]

    prediction = model.predict(X)[0]

    predicted_member = encoder.inverse_transform(
        [prediction]
    )[0]

    verified = predicted_member == expected_member_id

    return {
        "verified": bool(verified),
        "predicted_member_id": str(predicted_member),
        "expected_member_id": str(expected_member_id)
    }
