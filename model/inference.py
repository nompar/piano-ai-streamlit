import numpy as np
import tensorflow as tf

# Active AVANT tout chargement
tf.keras.config.enable_unsafe_deserialization()

model = None

def load_model(model_path="/Users/hadriendecaumont/Downloads/model1.keras"):
    """Charge le modèle une seule fois."""
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    return model


def predict(mel):
    """Fait l'inférence sur un mel spectrogramme."""
    model = load_model()

    mel_input = mel[np.newaxis, ...]
    onset_pred = model.predict(mel_input, verbose=0)[0]

    return onset_pred
