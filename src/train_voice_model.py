from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "audio_features.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_FILE)

feature_columns = [
    col for col in df.columns
    if col.startswith("mfcc_")
    or col in ["spectral_rolloff", "rms_energy"]
]

X = df[feature_columns]
y = df["member_id"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_DIR / "voice_model.pkl")
joblib.dump(encoder, MODEL_DIR / "voice_label_encoder.pkl")
joblib.dump(feature_columns, MODEL_DIR / "voice_feature_columns.pkl")

print("Model training completed!")
print("Training recordings:", len(X))
print("Features:", len(feature_columns))
print("Members:", list(encoder.classes_))
print("Scikit-learn version:", __import__("sklearn").__version__)
print("Model saved to:", MODEL_DIR)
