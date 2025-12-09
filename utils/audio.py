import os
import numpy as np
import librosa

# ====== Paramètres audio / spectrogramme ======
SR = 22000
HOP_LENGTH = 220
FPS = SR / HOP_LENGTH
N_FFT = 2048
N_MELS = 128


def audio_to_mel_3d(
    audio_path,
    sr=SR,
    hop_length=HOP_LENGTH,
    n_fft=N_FFT,
    n_mels=N_MELS,
    normalize=True,
):
    print(f"=== DEBUG audio_to_mel_3d ===")
    print(f"sr={sr}, hop_length={hop_length}, n_fft={n_fft}, n_mels={n_mels}")

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"Audio loaded - y.shape: {y.shape}, y.mean: {y.mean():.6f}")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    print(f"After melspectrogram - S.shape: {S.shape}, S.mean: {S.mean():.6f}")

    S_db = librosa.power_to_db(S, ref=np.max)
    print(f"After power_to_db - S_db.shape: {S_db.shape}, S_db.mean: {S_db.mean():.4f}")

    if normalize:
        S_min, S_max = S_db.min(), S_db.max()
        print(f"Before norm - S_min: {S_min:.4f}, S_max: {S_max:.4f}")
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
        print(f"After norm - S_db.mean: {S_db.mean():.4f}")

    mel_3d = S_db[..., np.newaxis]
    print(f"Final mel_3d.shape: {mel_3d.shape}, mel_3d.mean: {mel_3d.mean():.4f}")

    return mel_3d
