# !pip install -U openai-whisper
import whisper
import time
model = whisper.load_model("base")
start = time.time()
result = model.transcribe("cn2.mp3")
print(f"cost{time.time() - start}s")
print(result["text"])