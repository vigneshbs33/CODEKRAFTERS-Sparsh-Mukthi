# import sounddevice as sd
# from vosk import Model, KaldiRecognizer
# import queue
# import json
# import os

# # ===== Setup Vosk Model =====
# MODEL_PATH = "vosk-model-small-en-us-0.15"
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
# model = Model(MODEL_PATH)

# # ===== Audio Setup =====
# samplerate = 16000
# q = queue.Queue()

# def callback(indata, frames, time_info, status):
#     if status:
#         print(f"⚠️ {status}", flush=True)
#     q.put(bytes(indata))

# # ===== Recognizer Setup =====
# rec = KaldiRecognizer(model, samplerate)
# rec.SetWords(True)

# # ===== Main Transcription Loop =====
# print("📝 Live transcription started. Speak into the mic.")
# print("🔴 Press Ctrl+C to stop.\n")

# try:
#     with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
#                            channels=1, callback=callback):
#         while True:
#             data = q.get()
#             if rec.AcceptWaveform(data):
#                 result = json.loads(rec.Result())
#                 text = result.get("text", "")
#                 if text.strip():
#                     print("🗣️", text)
#             else:
#                 partial = json.loads(rec.PartialResult())
#                 partial_text = partial.get("partial", "")
#                 if partial_text.strip():
#                     print("...", partial_text, end="\r")

# except KeyboardInterrupt:
#     print("\n👋 Transcription stopped.")












import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue
import json
import os

# ===== Setup Vosk Model =====
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = Model(MODEL_PATH)

# ===== Audio Setup =====
samplerate = 16000
q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}", flush=True)
    q.put(bytes(indata))

# ===== Recognizer Setup =====
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(False)

# ===== Main Transcription Loop =====
print("📝 Real-time transcription started (Low Latency). Press Ctrl+C to stop.")

try:
    with sd.RawInputStream(samplerate=samplerate, blocksize=2048, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text.strip():
                    print("🗣️", text)
            else:
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text.strip():
                    print("...", partial_text, end='\r')

except KeyboardInterrupt:
    print("\n👋 Transcription stopped.")
