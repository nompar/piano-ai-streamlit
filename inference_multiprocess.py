import multiprocessing as mp

def run_inference(audio_path, midi_path, result_queue):
    """Fonction exécutée dans un processus isolé"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    import numpy as np
    import tensorflow as tf
    import librosa
    import pretty_midi

    SR = 22000
    HOP_LENGTH = 220
    N_FFT = 2048
    N_MELS = 128

    try:
        # Audio to mel
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_min, S_max = S_db.min(), S_db.max()
        S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
        mel_3d = S_db[..., np.newaxis]

        # Load model and predict
        model = tf.keras.models.load_model("./model_keras/model2.keras", compile=False)
        mel = np.squeeze(mel_3d, axis=-1).T
        mel_example = mel[np.newaxis, ...]
        onset_pred = model.predict(mel_example, verbose=0)[0]

        # Peak picking
        T, P = onset_pred.shape
        onset_binary = np.zeros_like(onset_pred, dtype=np.float32)
        for p in range(P):
            probs = onset_pred[:, p]
            t = 1
            last_t = -10
            while t < T - 1:
                if (probs[t] >= 0.85 and probs[t] >= probs[t-1] and
                    probs[t] >= probs[t+1] and t - last_t >= 10):
                    onset_binary[t, p] = 1.0
                    last_t = t
                    t += 10
                else:
                    t += 1

        # To MIDI
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        for p in range(onset_binary.shape[1]):
            t = 0
            while t < T:
                if onset_binary[t, p] >= 0.5:
                    start_frame = t
                    while t < T and onset_binary[t, p] >= 0.5:
                        t += 1
                    start_time = start_frame / 100
                    end_time = t / 100
                    if end_time - start_time < 0.25:
                        end_time = start_time + 0.25
                    piano.notes.append(pretty_midi.Note(velocity=40, pitch=21+p, start=start_time, end=end_time))
                else:
                    t += 1
        pm.instruments.append(piano)
        pm.write(midi_path)

        result_queue.put("OK")
    except Exception as e:
        result_queue.put(f"ERROR: {str(e)}")


def run_in_isolation(audio_path, midi_path):
    """Lance l'inférence dans un processus complètement isolé"""
    mp.set_start_method('spawn', force=True)

    result_queue = mp.Queue()
    p = mp.Process(target=run_inference, args=(audio_path, midi_path, result_queue))
    p.start()
    p.join(timeout=300)  # 5 minutes max

    if p.is_alive():
        p.terminate()
        return "ERROR: Timeout"

    return result_queue.get()
