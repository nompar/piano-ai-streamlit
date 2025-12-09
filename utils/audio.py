import numpy as np
import librosa

SR = 22000
HOP_LENGTH = 220
FPS = SR / HOP_LENGTH
N_FFT = 2048
N_MELS = 128


def audio_to_mel(audio_path):
    """Charge un audio et retourne le mel spectrogramme."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalisation
    S_min, S_max = S_db.min(), S_db.max()
    S_db = (S_db - S_min) / (S_max - S_min + 1e-8)

    # (n_mels, T) → (T, n_mels)
    mel = S_db.T.astype(np.float32)

    return mel
