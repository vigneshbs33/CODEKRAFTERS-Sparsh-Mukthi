import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import queue
import os
import sys

# Print Python version and sounddevice info
print(f"Python version: {sys.version}")
print(f"Sounddevice version: {sd.__version__}")

# Print available audio devices
print("\nAvailable audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")

# Check if Vosk model exists
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    sys.exit(1)
else:
    print(f"\nVosk model found at: {MODEL_PATH}")

# Initialize Vosk model
try:
    model = Model(MODEL_PATH)
    print("Vosk model loaded successfully")
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    sys.exit(1)

# Set up audio parameters
samplerate = 16000
q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    q.put(bytes(indata))

# Initialize recognizer
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(False)

print("\n=== Starting voice recognition test ===")
print("Speak into your microphone. Press Ctrl+C to stop.")

try:
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                          channels=1, callback=callback):
        print("Listening...")
        
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"Recognized: {text}")
            else:
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    print(f"Partial: {partial_text}", end="\r")
                    
except KeyboardInterrupt:
    print("\nTest stopped by user")
except Exception as e:
    print(f"\nError during test: {e}")
