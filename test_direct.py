import numpy as np
import tensorflow as tf
import librosa

SR = 22000
HOP_LENGTH = 220
N_FFT = 2048
N_MELS = 128

def audio_to_mel_3d(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
    return S_db[..., np.newaxis]

model = tf.keras.models.load_model("./model_keras/model2.keras", compile=False)

mel_3d = audio_to_mel_3d("/Users/hadriendecaumont/Desktop/2017_TEST/2017_test.mp3")
print(f"mel_3d mean: {mel_3d.mean():.4f}")

mel = np.squeeze(mel_3d, axis=-1).T
mel_example = mel[np.newaxis, ...]

onset_pred = model.predict(mel_example, verbose=0)[0]
print(f"onset_pred mean: {onset_pred.mean():.4f}")
