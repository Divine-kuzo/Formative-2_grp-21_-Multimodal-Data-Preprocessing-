"""
check_voice_csv.py -- does data/audio_features.csv match the recordings in audio/ ?

Run from the repo root:   python check_voice_csv.py
For every audio/<member>/<file>.wav it recomputes the 15 features and compares them with the
CSV rows that have the same member_id and filename.
  MATCH     = the CSV row equals the live features (max difference < 0.05)
  MISMATCH  = the CSV was made differently (other sample rate, trimming, normalisation, wrong label, ...)
"""
import numpy as np
import pandas as pd

from src.voice_fallback import CSV_PATH, FEATURE_COLS, extract_features, member_folders

df = pd.read_csv(CSV_PATH)
print(f"CSV: {df.shape[0]} rows, columns: {[c for c in df.columns if c not in FEATURE_COLS]}")
if "member_id" in df and "filename" in df:
    print(df.groupby(["member_id", "filename"]).size().to_string())
print()

sd = df[FEATURE_COLS].std().replace(0, 1).values
bad = 0
for folder in member_folders():
    for wav in sorted(folder.glob("*.wav")):
        live = extract_features(wav)[FEATURE_COLS].values[0]
        diffs = np.abs(df[FEATURE_COLS].values - live).max(axis=1)
        same = df[(df.member_id == folder.name) & (df.filename == wav.name)]
        d = np.linalg.norm((df[FEATURE_COLS].values - live) / sd, axis=1)
        j = int(d.argmin())
        best_same = diffs[same.index].min() if len(same) else None
        status = "NO CSV ROW" if best_same is None else ("MATCH" if best_same < 0.05 else "MISMATCH")
        bad += status != "MATCH"
        print(f"{folder.name}/{wav.name:<26} {status:<10} diff to its CSV row: "
              f"{'-' if best_same is None else format(best_same, '.2f'):<10} "
              f"closest CSV row overall: {df.member_id[j]}/{df.filename[j]}")
print(f"\n{bad} of the recordings do not match their CSV row.")