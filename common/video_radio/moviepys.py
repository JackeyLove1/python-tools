# pip install moviepy
from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    audio_clip.close()
    video_clip.close()

# Usage
extract_audio_from_video('test.mp4', 'audio.wav')

import whisper
import time
model = whisper.load_model("small")
start = time.time()
result = model.transcribe('audio.wav')
print(f"cost{time.time() - start}s")
print(result["text"])