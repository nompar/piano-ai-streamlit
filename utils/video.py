import os
import shutil
from synthviz import create_video


def generate_video(midi_path, video_path):
    """Génère une vidéo et nettoie les fichiers temporaires."""
    create_video(
        input_midi=midi_path,
        video_filename=video_path,
        image_width=1280,
        image_height=720,
        fps=30,
        falling_note_color=[75, 105, 177],
        pressed_key_color=[75, 105, 177],
    )

    # Cleanup des fichiers créés par synthviz
    if os.path.exists("video_frames"):
        shutil.rmtree("video_frames")
