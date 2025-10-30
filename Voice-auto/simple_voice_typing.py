import speech_recognition as sr
import pyautogui
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def type_text(text):
    """Type the recognized text"""
    if not text.strip():
        return True
        
    # Special commands
    if text.lower() in ["stop typing", "exit typing", "quit"]:
        logger.info("Received command to stop typing")
        return False
    elif text.lower() in ["delete", "backspace"]:
        pyautogui.press('backspace')
        logger.info("Executed command: backspace")
    elif text.lower() in ["enter", "new line"]:
        pyautogui.press('enter')
        logger.info("Executed command: enter")
    elif text.lower() in ["tab", "indent"]:
        pyautogui.press('tab')
        logger.info("Executed command: tab")
    elif text.lower() in ["space"]:
        pyautogui.press('space')
        logger.info("Executed command: space")
    elif text.lower() in ["clear all", "clear"]:
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('delete')
        logger.info("Executed command: clear all")
    else:
        # Type the text character by character
        logger.info(f"Typing text: {text}")
        pyautogui.write(text + " ")
        
    return True

def main():
    # Initialize recognizer
    r = sr.Recognizer()
    r.energy_threshold = 300  # Adjust based on your microphone sensitivity
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8
    
    print("📝 Voice typing started. Speak into your microphone.")
    print("🔴 Say 'stop typing' or press Ctrl+C to stop.\n")
    
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            print("Calibrating for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Calibration complete. Start speaking!")
            
            logger.info("Voice typing activated - speak to type text")
            
            continue_typing = True
            while continue_typing:
                try:
                    # Listen for audio
                    print("Listening...")
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    
                    # Recognize speech
                    print("Processing...")
                    text = r.recognize_google(audio)
                    
                    if text:
                        print(f"🗣️ Recognized: {text}")
                        continue_typing = type_text(text)
                        
                except sr.WaitTimeoutError:
                    print("No speech detected. Still listening...")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Could not request results; {e}")
                    print(f"Error with speech recognition service: {e}")
                    
    except KeyboardInterrupt:
        logger.info("Voice typing stopped by keyboard interrupt")
        print("\n👋 Voice typing stopped.")
    except Exception as e:
        logger.error(f"Error in voice typing: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
