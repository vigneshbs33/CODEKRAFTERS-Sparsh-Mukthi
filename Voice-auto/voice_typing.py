import speech_recognition as sr
import pyautogui
import time
import logging
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to control the voice typing loop
running = True

# Initialize the recognizer
recognizer = sr.Recognizer()

# Adjust for ambient noise and sensitivity
recognizer.energy_threshold = 300  # Increase if it's too sensitive
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Seconds of silence before considering the phrase complete

# Function to type the recognized text
def type_text(text):
    if text.strip():
        # Special commands
        if text.lower() in ["delete", "backspace"]:
            pyautogui.press('backspace')
            logger.info(f"Executed command: backspace")
        elif text.lower() in ["enter", "new line"]:
            pyautogui.press('enter')
            logger.info(f"Executed command: enter")
        elif text.lower() in ["tab", "indent"]:
            pyautogui.press('tab')
            logger.info(f"Executed command: tab")
        elif text.lower() in ["space"]:
            pyautogui.press('space')
            logger.info(f"Executed command: space")
        elif text.lower() in ["clear all", "clear"]:
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            logger.info(f"Executed command: clear all")
        elif text.lower() in ["stop typing", "exit typing"]:
            logger.info(f"Received command to stop typing")
            return False
        else:
            # Type the text directly without adding space (will be added by the recognizer)
            logger.info(f"Typing text: {text}")
            # Use typewrite instead of write for better compatibility
            for char in text:
                pyautogui.typewrite(char)
            # Add a space at the end
            pyautogui.typewrite(' ')
    return True

# Main voice typing function
def voice_typing_loop():
    global running
    
    print("📝 Voice typing started. Speak into the mic.")
    print("🔴 Say 'stop typing' or press Ctrl+C to stop.\n")
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise once at the beginning
        print("Calibrating for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Calibration complete. Start speaking!")
        
        logger.info("Voice typing activated - speak to type text")
        
        while running:
            try:
                # Listen for audio input
                print("Listening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                
                # Try to recognize the speech
                print("Processing...")
                text = recognizer.recognize_google(audio)
                
                if text:
                    print(f"🗣️ Recognized: {text}")
                    # Process the recognized text
                    if not type_text(text):
                        logger.info("Voice typing stopped by voice command")
                        running = False
                        break
                    
            except sr.WaitTimeoutError:
                print("No speech detected. Still listening...")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Could not request results; {e}")
                print(f"Error with speech recognition service: {e}")
            except Exception as e:
                logger.error(f"Error in voice typing: {str(e)}")
                print(f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    try:
        # Start the voice typing in a thread
        voice_thread = threading.Thread(target=voice_typing_loop)
        voice_thread.daemon = True
        voice_thread.start()
        
        # Main thread waits for keyboard interrupt
        while voice_thread.is_alive():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        # Set the flag to stop the voice typing thread
        running = False
        logger.info("Voice typing stopped by keyboard interrupt")
        print("\n👋 Voice typing stopped.")
        
        # Wait for the thread to finish
        voice_thread.join(timeout=2)
    except Exception as e:
        running = False
        logger.error(f"Error in main thread: {str(e)}")
        print(f"Error: {str(e)}")
