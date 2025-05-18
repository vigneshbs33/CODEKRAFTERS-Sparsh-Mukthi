import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3
import keyboard
import time
import os

# Load command dictionary
with open("commands.json", "r") as f:
    command_map = json.load(f)

# Invert to keyword → action map
keyword_to_action = {}
for action, variants in command_map.items():
    for word in variants:
        keyword_to_action[word.lower()] = action.lower()

# Text-to-speech
engine = pyttsx3.init()
def announce_action(action):
    engine.say(f"{action} key pressed")
    engine.runAndWait()

# Load Vosk model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = Model(MODEL_PATH)

# Constants
samplerate = 16000
duration = 2  # seconds to listen each cycle
cooldown = 1.5
last_triggered = {}

print("🎙️ Ready. Say a command (Ctrl+C to exit).")




# def listen_once():
#     """Captures audio for a short duration and returns recognized text."""
#     rec = KaldiRecognizer(model, samplerate)
#     rec.SetWords(False)

#     with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
#                            channels=1) as stream:
#         print("🎧 Listening...")
#         audio = b""
#         start = time.time()
#         while time.time() - start < duration:
#             data, overflowed = stream.read(4000)
#             if overflowed:
#                 print("⚠️ Overflow!")
#             audio += bytes(data)
#             if rec.AcceptWaveform(bytes(data)):
#                 break

#         if rec.AcceptWaveform(audio):
#             result = rec.Result()
#         else:
#             result = rec.FinalResult()

#     return json.loads(result).get("text", "").strip().lower()




def listen_once():
    """Captures audio for a short duration and returns recognized text."""
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(False)

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1) as stream:
        print("🎧 Listening...")
        audio = b""
        start = time.time()
        while time.time() - start < duration:
            data, overflowed = stream.read(4000)
            if overflowed:
                print("⚠️ Overflow!")
            audio += bytes(data)
            if rec.AcceptWaveform(bytes(data)):
                break

        if rec.AcceptWaveform(audio):
            result = rec.Result()
        else:
            result = rec.FinalResult()

    return json.loads(result).get("text", "").strip().lower()






# Infinite loop
try:
    while True:
        text = listen_once()
        if not text:
            print("🤔 No command detected.")
            continue

        print(f"🗣️ Detected: {text}")
        words = text.split()
        now = time.time()

        for word in words:
            if word in keyword_to_action:
                action = keyword_to_action[word]
                last_time = last_triggered.get(action, 0)
                if now - last_time >= cooldown:
                    print(f"✅ Matched → {word} → {action}")
                    key = 'space' if action == 'jump' else action
                    keyboard.press(key)
                    time.sleep(0.2)
                    keyboard.release(key)
                    announce_action(action)
                    last_triggered[action] = now
                break

except KeyboardInterrupt:
    print("\n👋 Exiting. Goodbye!")
