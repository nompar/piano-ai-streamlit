import numpy as np
import pretty_midi

PITCH_MIN = 21  # A0
N_PITCHES = 88  # 21–108
FPS = 100       # comme dans le preprocess


def probs_to_onset_binary(onset_pred, threshold=0.50, min_distance=10):
    """
    onset_pred : (T, 88) proba ∈ [0,1]
    threshold  : seuil proba
    min_distance : nombre de frames minimum entre 2 onsets du même pitch
    """
    T, P = onset_pred.shape
    onset_binary = np.zeros_like(onset_pred, dtype=np.float32)

    for p in range(P):
        probs = onset_pred[:, p]
        t = 1
        last_t = -min_distance

        while t < T - 1:
            # pic local : plus grand que ses voisins
            if (
                probs[t] >= threshold and
                probs[t] >= probs[t - 1] and
                probs[t] >= probs[t + 1] and
                t - last_t >= min_distance
            ):
                onset_binary[t, p] = 1.0
                last_t = t
                t += min_distance  # on saute un peu pour éviter les clusters
            else:
                t += 1

    return onset_binary


def onset_binary_to_midi(onset_binary, output_path, fps=FPS, pitch_min=PITCH_MIN, min_duration=0.25, velocity=40):
    """
    onset_binary : np.ndarray (T, 88) de 0/1
    fps          : frames par seconde (ici 100)
    """
    T, n_pitches = onset_binary.shape
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

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

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch_min + p,
                    start=start_time,
                    end=end_time,
                )
                piano.notes.append(note)
            else:
                t += 1

    pm.instruments.append(piano)
    pm.write(output_path)
