import streamlit as st
import tempfile
import os
from inference_multiprocess import run_in_isolation
from utils.video import generate_video

st.set_page_config(page_title="Piano AI", page_icon="🎹")
st.title("🎹 Piano AI")

audio_file = st.file_uploader("Upload piano audio", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file)

    if st.button("🎵 Transcribe", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "input.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

            midi_path = os.path.join(tmp, "output.mid")

            progress = st.progress(0, text="Processing...")
            progress.progress(30, text="Running model...")

            result = run_in_isolation(audio_path, midi_path)

            if result.startswith("ERROR"):
                st.error(result)
            else:
                progress.progress(70, text="Creating video...")
                video_path = os.path.join(tmp, "output.mp4")
                generate_video(midi_path, video_path)

                progress.progress(100, text="Done!")
                st.success("Transcription complete!")

                st.video(video_path)

                col1, col2 = st.columns(2)
                with col1:
                    with open(midi_path, "rb") as f:
                        st.download_button("⬇️ MIDI", f.read(), "transcription.mid")
                with col2:
                    with open(video_path, "rb") as f:
                        st.download_button("⬇️ Video", f.read(), "piano_roll.mp4")
