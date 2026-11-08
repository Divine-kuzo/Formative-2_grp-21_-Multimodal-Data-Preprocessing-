from pathlib import Path
import pandas as pd

from src.audio_utils import extract_audio_features

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "audio_features.csv"

records = []

for member_folder in sorted(AUDIO_DIR.iterdir()):
    if not member_folder.is_dir():
        continue

    for audio_file in sorted(member_folder.glob("*.wav")):
        print(f"Processing: {member_folder.name}/{audio_file.name}")

        features = extract_audio_features(audio_file)
        features["member_id"] = member_folder.name
        features["filename"] = audio_file.name

        records.append(features)

if not records:
    raise RuntimeError("No WAV recordings found.")

df = pd.DataFrame(records)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset created successfully!")
print("Shape:", df.shape)
print("\nRecordings per member:")
print(df["member_id"].value_counts().sort_index())
print("\nSaved to:", OUTPUT_FILE)
