import numpy as np
import tensorflow as tf

model = None

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

def load_model(model_path="./model_keras/model2.keras"):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model name: {model.name}")
        for i, layer in enumerate(model.layers):
            w = layer.get_weights()
            if w:
                print(f"Layer {i} ({layer.name}) weights mean: {w[0].mean():.6f}")
                break
    return model

def predict(mel_3d):
    model = load_model()

    print(f"=== DEBUG predict ===")
    print(f"mel_3d input - shape: {mel_3d.shape}, mean: {mel_3d.mean():.4f}")

    mel = np.squeeze(mel_3d, axis=-1).T
    print(f"After squeeze+T - shape: {mel.shape}, mean: {mel.mean():.4f}")

    mel_example = mel[np.newaxis, ...]
    print(f"After newaxis - shape: {mel_example.shape}, mean: {mel_example.mean():.4f}")

    # Force le mode inference explicitement
    onset_pred = model(mel_example, training=False)[0].numpy()
    print(f"onset_pred - shape: {onset_pred.shape}, mean: {onset_pred.mean():.4f}")

    return onset_pred
