import streamlit as st
import tempfile
import os

from utils.audio import audio_to_mel
from utils.midi import probs_to_onset_binary, onset_to_midi
from utils.video import generate_video
from model.inference import predict

st.set_page_config(page_title="Piano AI", page_icon="🎹")

st.title("🎹 Piano AI")
st.write("Upload a piano audio file to generate a piano roll video.")

audio_file = st.file_uploader("Upload piano audio", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file)

    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5)

    if st.button("🎵 Transcribe", type="primary"):

        with tempfile.TemporaryDirectory() as tmp:
            # Sauvegarder l'audio
            audio_path = os.path.join(tmp, "input.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

            # Progress bar
            progress = st.progress(0, text="Processing...")

            # 1. Préprocessing
            progress.progress(20, text="Extracting mel spectrogram...")
            mel = audio_to_mel(audio_path)

            # 2. Inférence
            progress.progress(40, text="Running model...")
            onset_pred = predict(mel)

            # 3. Post-processing
            progress.progress(60, text="Generating MIDI...")
            onset_binary = probs_to_onset_binary(onset_pred, threshold=threshold)
            midi_path = os.path.join(tmp, "output.mid")
            onset_to_midi(onset_binary, midi_path)

            # 4. Génération vidéo
            progress.progress(80, text="Creating video...")
            video_path = os.path.join(tmp, "output.mp4")
            generate_video(midi_path, video_path)

            progress.progress(100, text="Done!")

            # Afficher résultats
            st.success("Transcription complete!")

            st.subheader("📹 Piano Roll Video")
            st.video(video_path)

            # Boutons download
            col1, col2 = st.columns(2)

            with col1:
                with open(midi_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download MIDI",
                        f.read(),
                        "transcription.mid",
                        mime="audio/midi"
                    )

            with col2:
                with open(video_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Video",
                        f.read(),
                        "piano_roll.mp4",
                        mime="video/mp4"
                    )
