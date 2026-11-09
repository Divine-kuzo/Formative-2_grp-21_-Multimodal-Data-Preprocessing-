import numpy as np, pandas as pd
from src.audio_utils import extract_audio_features, load_voice_artifacts

model, cols = load_voice_artifacts()
print(model)
df = pd.read_csv("data/audio_features.csv")
print(df.shape, df.member_id.value_counts().to_dict())

wav = "audio/member_01/yes_approve.wav"
live = extract_audio_features(wav)
rows = df[(df.member_id == "member_01") & (df.filename.str.contains("yes_approve"))]
print("matching CSV rows:", len(rows))
for _, row in rows.iterrows():
    diff = max(abs(live[c] - row[c]) for c in cols)
    csv_pred = model.predict(np.asarray([[row[c] for c in cols]], dtype=float))[0]
    print(row.filename, "| max feature difference:", round(diff, 3), "| model on CSV row:", csv_pred)
print("model on live features:", model.predict(np.asarray([[live[c] for c in cols]], dtype=float))[0])