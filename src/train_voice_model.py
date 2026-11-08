from pathlib import Path
import tempfile

import joblib
import librosa
import numpy as np
import soundfile as sf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.audio_utils import extract_audio_features

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    *[f"mfcc_{i}" for i in range(1, 14)],
    "spectral_rolloff",
    "rms_energy"
]

X_train = []
y_train = []
X_test = []
y_test = []

with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)

    for member_dir in sorted(AUDIO_DIR.glob("member_*")):
        if not member_dir.is_dir():
            continue

        member_id = member_dir.name
        train_file = member_dir / "confirm_transaction.wav"
        test_file = member_dir / "yes_approve.wav"

        if not train_file.is_file() or not test_file.is_file():
            raise FileNotFoundError(
                f"Missing training or testing recording for {member_id}"
            )

        # Original training recording
        audio, sr = librosa.load(train_file, sr=None, mono=True)

        training_versions = [
            ("original", audio),
            ("pitch_shift", librosa.effects.pitch_shift(
                y=audio, sr=sr, n_steps=2
            )),
            ("time_stretch", librosa.effects.time_stretch(
                y=audio, rate=1.1
            ))
        ]

        for version_name, augmented_audio in training_versions:
            temp_file = temp_dir / f"{member_id}_{version_name}.wav"
            sf.write(temp_file, augmented_audio, sr)

            features = extract_audio_features(temp_file)

            X_train.append([
                features[column] for column in FEATURE_COLUMNS
            ])
            y_train.append(member_id)

        # Untouched test recording
        test_features = extract_audio_features(test_file)

        X_test.append([
            test_features[column] for column in FEATURE_COLUMNS
        ])
        y_test.append(member_id)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(class_weight="balanced"))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n========== EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test, predictions, labels=sorted(set(y_test))
))

print("\nActual:", y_test)
print("Predicted:", predictions.tolist())

# Save separately: do not overwrite the existing working model.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, OUTPUT_DIR / "voice_model.pkl")
joblib.dump(FEATURE_COLUMNS, OUTPUT_DIR / "voice_feature_columns.pkl")

print("\nFinal SVM model saved to:", OUTPUT_DIR)
