import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'


import sys
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

def probs_to_onset_binary(onset_pred, threshold=0.85, min_distance=10):
    T, P = onset_pred.shape
    onset_binary = np.zeros_like(onset_pred, dtype=np.float32)
    for p in range(P):
        probs = onset_pred[:, p]
        t = 1
        last_t = -min_distance
        while t < T - 1:
            if (probs[t] >= threshold and probs[t] >= probs[t - 1] and
                probs[t] >= probs[t + 1] and t - last_t >= min_distance):
                onset_binary[t, p] = 1.0
                last_t = t
                t += min_distance
            else:
                t += 1
    return onset_binary

def onset_binary_to_midi(onset_binary, output_path, fps=100, pitch_min=21, min_duration=0.25, velocity=40):
    import pretty_midi
    T, n_pitches = onset_binary.shape
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for p in range(n_pitches):
        t = 0
        while t < T:
            if onset_binary[t, p] >= 0.5:
                start_frame = t
                while t < T and onset_binary[t, p] >= 0.5:
                    t += 1
                end_frame = t
                start_time = start_frame / fps
                end_time = end_frame / fps
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration
                note = pretty_midi.Note(velocity=velocity, pitch=pitch_min + p, start=start_time, end=end_time)
                piano.notes.append(note)
            else:
                t += 1
    pm.instruments.append(piano)
    pm.write(output_path)

if __name__ == "__main__":
    audio_path = sys.argv[1]
    midi_output = sys.argv[2]

    model = tf.keras.models.load_model("./model_keras/model2.keras", compile=False)

    mel_3d = audio_to_mel_3d(audio_path)
    mel = np.squeeze(mel_3d, axis=-1).T
    mel_example = mel[np.newaxis, ...]

    onset_pred = model.predict(mel_example, verbose=0)[0]
    onset_binary = probs_to_onset_binary(onset_pred)
    onset_binary_to_midi(onset_binary, midi_output)

    print("OK")
