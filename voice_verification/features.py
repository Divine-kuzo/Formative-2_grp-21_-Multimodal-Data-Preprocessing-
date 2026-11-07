
import numpy as np
import librosa

SAMPLE_RATE = 22050

def extract_features(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features = {}

    for i in range(13):
        features[f"mfcc_{i+1}"] = float(
            np.mean(mfcc[i])
        )

    features["spectral_rolloff"] = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr
            )
        )
    )

    features["rms_energy"] = float(
        np.mean(
            librosa.feature.rms(y=audio)
        )
    )

    return features
