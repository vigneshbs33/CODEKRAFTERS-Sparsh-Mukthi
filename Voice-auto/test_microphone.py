import sounddevice as sd
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_sound_devices():
    """Print all available sound devices"""
    devices = sd.query_devices()
    logger.info("Available sound devices:")
    for i, device in enumerate(devices):
        logger.info(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")

def test_microphone(duration=5, device=None):
    """Test microphone by recording audio and checking if sound is detected"""
    logger.info(f"Testing microphone for {duration} seconds...")
    
    # Parameters
    samplerate = 16000
    channels = 1
    
    # Record audio
    recording = sd.rec(
        int(samplerate * duration), 
        samplerate=samplerate, 
        channels=channels,
        device=device
    )
    
    logger.info("Recording... Speak into your microphone")
    
    # Show a simple level meter while recording
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Get the current recording level
            if len(recording) > 0:
                current_frame = int((time.time() - start_time) * samplerate)
                if current_frame < len(recording):
                    level = np.abs(recording[:current_frame]).max() if current_frame > 0 else 0
                    meter = "#" * int(level * 50)
                    print(f"Level: {meter}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    # Wait for recording to complete
    sd.wait()
    
    # Analyze the recording
    max_amplitude = np.abs(recording).max()
    logger.info(f"Max amplitude detected: {max_amplitude:.6f}")
    
    if max_amplitude > 0.01:
        logger.info("✅ Microphone is working! Sound detected.")
    else:
        logger.info("❌ No significant sound detected. Check your microphone settings.")
    
    return max_amplitude > 0.01

if __name__ == "__main__":
    print_sound_devices()
    
    # Test the default microphone
    if test_microphone():
        logger.info("Microphone test passed successfully.")
    else:
        logger.info("Microphone test failed. Please check your microphone settings.")
