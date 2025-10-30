<div align="center">
  <h1>✨ Sparsh Mukthi - Advanced Gesture & Voice Control System</h1>
  
  <div align="center" style="margin: 20px 0;">
    <a href="https://sparsh-mukthi.vercel.app/" target="_blank">
      <img src="https://img.shields.io/badge/🚀-Live%20Demo-4CAF50?style=for-the-badge&logo=vercel&logoColor=white&labelColor=1a1a2e" alt="Live Demo" />
      <img src="https://img.shields.io/badge/⚡-Try%20Now!-FF6B6B?style=for-the-badge&logo=vercel&logoColor=white&labelColor=1a1a2e" alt="Try Now" />
    </a>
    <p style="margin: 10px 0; font-size: 0.9em; color: #a8a8a8;">
      <i class="fas fa-external-link-alt"></i> Experience the future of touchless interaction
    </p>
    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
      <a href="#%EF%B8%8F-getting-started" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/📖-Documentation-6c63ff?style=flat-square" alt="Documentation" />
      </a>
      <a href="#-adaptive-ai-in-action" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/✨-Features-6c63ff?style=flat-square" alt="Features" />
      </a>
    </div>
  </div>
  
  <img src="https://i.ibb.co/217zjLmn/temp.webp" alt="Sparsh Mukthi Logo" width="200"/>
  
  <p><i>Touchless Control System for Education, Healthcare, and VR Gaming</i></p>
  <p><b>Cross-Platform Solution for Windows, macOS, and Linux</b></p>
</div>

## 🌟 Overview

Sparsh Mukthi is an advanced gesture and voice control system that revolutionizes human-computer interaction. Built with cutting-edge computer vision and machine learning technologies, it offers a seamless touchless experience across multiple domains including Education, Healthcare, and Gaming.

### 🚀 Key Features

- **Smart Gesture Recognition**
  - Real-time hand tracking with MediaPipe
  - Custom gesture training with AI-assisted learning
  - Multiple gesture support with confidence scoring
  - Performance optimization based on usage patterns
  - Visual and audio feedback for better interaction

- **Dual Control Modes**
  - **Education/Healthcare Mode (edu-hcare.py)**:
    - Precise cursor control for professional environments
    - Touchless interaction for sterile healthcare settings
    - Presentation controls for educators
    - Virtual anatomy manipulation
    - Accessibility focus for different ability levels
  
  - **Gaming/VR Mode (air-control.py)**:
    - Optimized for fast-paced interactions
    - Natural VR navigation and controls
    - 3D space manipulation
    - Advanced gesture combinations
    - Smooth motion tracking for immersive experiences

- **Voice Integration**
  - Dual-mode voice control system:
    - **Command Mode**: Execute keyboard commands by voice
    - **Typing Mode**: Transcribe speech to text at cursor position
  - Natural language commands with customizable variants
  - Voice-gesture hybrid control for hands-free operation
  - Noise-resistant recognition using Vosk
  - Real-time voice feedback and status indicators

- **Adaptive AI Technology**
  - Machine learning for continuous improvement
  - Personalized user profiles and preferences
  - Context-aware processing for different applications
  - Feedback loop that learns from corrections
  - Usage analytics to optimize command recognition

## 🤖 Adaptive AI in Action

The application's adaptive AI capabilities enable it to:

1. **Learn and Improve**
   - Tracks successful recognitions and corrections
   - Adjusts recognition thresholds in real-time
   - Adapts to different lighting conditions and environments

2. **Personalized Experience**
   - Creates unique user profiles
   - Remembers preferred gestures and commands
   - Adjusts sensitivity based on interaction patterns

3. **Context-Aware Processing**
   - Detects the current application context
   - Adjusts gesture recognition parameters accordingly
   - Optimizes performance for different use cases

## 🎯 Use Cases

### 1. Education Sector
- **Virtual Labs**: Touchless control of virtual experiments
- **3D Learning**: Manipulate 3D models in space
- **Interactive Presentations**: Control slides and demos with gestures
- **Distance Learning**: Enhanced remote instruction capabilities
- **Special Needs**: Adaptive learning interfaces

### 2. Healthcare Applications
- **Sterile Environments**: Touch-free computer control in operating rooms
- **Patient Care**: Contactless patient data access
- **Medical Imaging**: Gesture-based image manipulation
- **Rehabilitation**: Interactive physical therapy exercises
- **Infection Control**: COVID-safe workplace interactions

### 3. VR and Gaming
- **VR Navigation**: Natural movement in virtual spaces
- **Game Control**: Precise gesture-based commands
- **3D Modeling**: Intuitive object manipulation
- **Virtual Training**: Immersive simulation control
- **Fitness Games**: Body movement tracking

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam with at least 640x480 resolution
- Microphone (for voice features)
- Modern web browser (Chrome, Firefox, or Edge recommended)
- VR headset (optional, for VR features)

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vigneshbs33/CODEKRAFTERS-Sparsh-Mukthi
   cd CODEKRAFTERS-Sparsh-Mukthi
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install System Dependencies**:

   #### Windows
   - Install Microsoft Visual C++ Redistributable (required for some Python packages)
   - Ensure camera and microphone permissions are enabled in Windows Settings → Privacy

   #### Linux
   ```bash
   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y \
       python3-opencv \
       python3-tk \
       python3-dev \
       portaudio19-dev \
       ffmpeg \
       libx11-dev \
       libxtst-dev

   # Set up device permissions
   sudo usermod -a -G video $USER
   sudo usermod -a -G audio $USER
   ```

   #### macOS
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install dependencies
   brew install portaudio ffmpeg

   # For GUI support (if needed)
   brew install --cask xquartz
   ```

5. **Download the Vosk Speech Recognition Model**:
   ```bash
   python Voice-auto/download_vosk_model.py
   ```

   Or download manually from [Vosk Models](https://alphacephei.com/vosk/models) and extract the `vosk-model-small-en-us-0.15` folder to the `Voice-auto` directory.

## 🚀 Getting Started

### Running the Application

1. After installation, launch the web interface:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Grant camera and microphone permissions when prompted

### Using the Interface

The web interface provides access to all features through a modern, intuitive UI:

- **Navigation Panel**: Access different features using the cyberpunk-styled sidebar
- **Controller Buttons**: Start and stop various modes with one click
- **Status Indicators**: Real-time feedback on system status and active features
- **Gesture Training**: Custom gesture teaching interface with visual feedback

### API Endpoints

The application provides RESTful API endpoints for controller management:

- `/start_mouse_controller`: Starts the Education/Healthcare controller (supports both GET/POST)
- `/start_air_controller`: Starts the Gaming/VR controller (supports both GET/POST)
- `/stop_process`: Safely terminates a running controller
- `/get_gestures`: Retrieves available gestures and their mappings
- `/update_mapping`: Updates key mappings for gestures

## 🎮 Using Sparsh Mukthi

### Gesture Training

1. Click "Start Training" in the web interface
2. Select a gesture name from the dropdown or create a new one
3. Perform the gesture in front of your webcam
4. The system will capture multiple samples
5. Train the model when prompted

### Voice Commands

1. Start the Voice Command mode
2. Use the following default commands or customize them:
   - "Scroll up/down" - Scroll the page
   - "Click" - Perform a mouse click
   - "Type [text]" - Type the specified text
   - "Open [application]" - Launch an application

### Control Modes

#### Education/Healthcare Mode
- Precise cursor control
- Right/left click gestures
- Scrolling and zooming
- Ideal for presentations and medical imaging

#### Gaming Mode
- Fast response controls
- Customizable key bindings
- Gesture combinations
- Optimized for gaming performance

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working
- Ensure no other application is using the webcam
- Check camera permissions in system settings
- Try a different USB port if using an external camera

#### Voice Recognition Issues
- Ensure microphone is properly connected
- Reduce background noise
- Check microphone volume levels
- Retrain voice commands if needed

#### Performance Problems
- Close other resource-intensive applications
- Reduce webcam resolution if needed
- Ensure your system meets the minimum requirements

## 📊 System Requirements

### Minimum
- OS: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- CPU: Intel i3 or equivalent
- RAM: 4GB
- Webcam: 720p or higher
- Python: 3.8+

### Recommended
- OS: Windows 11, macOS 12+, or Ubuntu 20.04+
- CPU: Intel i5 or equivalent
- RAM: 8GB or more
- Webcam: 1080p with autofocus
- Dedicated GPU for better performance

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- Vosk for speech recognition
- OpenCV for computer vision
- The open-source community for their invaluable contributions

### Available Features

1. **Gesture Training & Recognition**
   - Train custom gestures with your webcam
   - Map gestures to keyboard commands
   - Use gestures to control applications

2. **Mouse Controllers**
   - **Education & Healthcare Mode** (EDU-HCARE): Precise cursor control for educational and medical applications
   - **Air Controller**: Gaming-optimized controller for FPS and other games

3. **Voice Control System**
   - **Voice Commands**: Control your computer with voice commands mapped to keyboard shortcuts
   - **Voice Typing**: Dictate text directly to any text field

4. **Demo Experiences**
   - **Education Demo**: Try the EDU-HCARE controller with 3D anatomy models
   - **Gaming Demo**: Experience the Air Controller with a browser-based FPS game

### System Architecture

The application consists of several interconnected components:

- **Main Web Interface** (`app.py`): Flask server providing the UI and controlling all subsystems
  - RESTful API endpoints for controller management
  - WebSocket support for real-time updates
  - Process lifecycle management
- **Gesture Recognition** (`gesture_data/adaptive-gesture-ai/`): Custom gesture training and recognition
- **Mouse Controllers** (`Main-flow-gesture/`): 
  - `edu-hcare.py`: Precision mouse control for education/medical use
  - `air-controller.py`: Specialized controller for gaming
- **Voice System** (`Voice-auto/`): Voice command and dictation capabilities

#### Voice Command Mode
- Speak commands to execute keyboard actions (e.g., "left", "right", "up", "down")
- Commands are processed when cursor is not in a text field
- Customizable command variants in commands.json
- Provides audio feedback when commands are recognized

#### Voice Typing Mode
- Speak naturally to type text where your cursor is positioned
- Works in any text field or text editor
- Special commands available:
  - "delete" or "backspace": Deletes the last character
  - "enter" or "new line": Creates a new line
  - "tab" or "indent": Inserts a tab
  - "space": Inserts a space
  - "clear all" or "clear": Clears all text
  - "stop typing" or "exit typing": Stops voice typing mode

4. **Custom AI Gestures**:
```bash
python app.py
```
- Access web interface at http://127.0.0.1:5000

```
sparsh-mukthi/
├─ app.py                    # Main Flask application with all controller routes
├─ requirements.txt          # Project dependencies
├─ Main-flow-gesture/        # Specialized mouse controllers
│   ├─ edu-hcare.py          # Education/Healthcare precision mouse controller
│   └─ air-controller.py     # Air gesture controller optimized for gaming
├─ Voice-auto/              # Voice control system
│   ├─ final-model.py        # Voice command recognition module
│   ├─ commands.json         # Voice command mappings configuration
│   └─ live_transcriber.py   # Real-time speech-to-text transcription
├─ gesture_data/            # Gesture recognition system
│   └─ adaptive-gesture-ai/  # Custom gesture training and prediction
│       ├─ collect_gestures.py # Gesture training module
│       ├─ predict_live.py     # Real-time gesture recognition
│       ├─ custom_gestures.json # Trained gesture database
│       └─ key_mappings.json   # Gesture-to-keyboard mapping
├─ static/                  # Web interface assets
├─ templates/               # HTML templates for web interface
└─ logs/                    # Application logs
```

### Core Components

1. **Main Application (`app.py`)**
   - Flask web server handling all HTTP routes
   - Process management for all controllers
   - User interface rendering
   - API endpoints for all functionalities

2. **Mouse Controllers**
   - **EDU-HCARE** (`edu-hcare.py`): Education and healthcare-oriented mouse controller with precise movements
   - **Air Controller** (`air-controller.py`): Gaming-optimized controller with special gestures for FPS games

3. **Voice System**
   - Command mode for executing keyboard shortcuts
   - Transcription mode for dictation
   - Vosk-based speech recognition

4. **Gesture Recognition**
   - Custom gesture training interface
   - Real-time gesture prediction
   - Keyboard/mouse action mapping

### Technologies Used

- **Core**: Python 3.8+, OpenCV 4.8+
- **AI/ML**: MediaPipe, NumPy, Scikit-learn, Joblib
- **VR Integration**: PyGame, AutoPy
- **Voice Processing**: Vosk, SoundDevice, Pyttsx3, Keyboard
- **Web Interface**: Flask, Flask-SocketIO
- **Input Control**: PyAutoGUI, Pynput
- **System Integration**: psutil (for process management), subprocess (for controller execution)

## 🔄 Cross-Platform Compatibility

Sparsh Mukthi is designed to work across Windows, macOS, and Linux with minimal configuration differences. Here's how to ensure compatibility on each platform:

### Windows

- **Default Configuration**: Works out of the box on Windows 10/11
- **Process Management**: Uses Windows-specific process creation flags to avoid command prompt windows
- **Camera Access**: Automatically detects and uses the default camera
- **Dependencies**: All required libraries are installed via pip

### macOS

- **Permission Requirements**: Requires explicit permissions for camera, microphone, and accessibility
- **Dependency Management**: Some libraries (like portaudio) may require Homebrew installation
- **Process Management**: Uses POSIX-compliant process handling
- **Keyboard/Mouse Control**: Requires accessibility permissions to be granted in System Preferences

### Linux

- **Distribution Support**: Tested on Ubuntu 20.04+ and Debian-based distributions
- **System Dependencies**: Requires X11 libraries, OpenCV dependencies, and audio libraries
- **Device Access**: Requires appropriate user group permissions (video, audio)
- **Display Management**: May require X11 forwarding configuration in some environments

## 📦 First-Run Setup

After cloning the repository and installing dependencies, follow these steps:

1. **Verify Installation**:
   ```bash
   # Check if all required packages are installed
   pip list | grep -E "opencv|numpy|mediapipe|flask|vosk"
   ```

2. **Initial Configuration**:
   ```bash
   python setup_check.py
   ```
   This will verify your system compatibility and set up any necessary configurations.

3. **Path Configuration**:
   Ensure all paths in the application are relative to the project root. If you encounter path-related errors, check:
   - File paths in app.py
   - Resource paths in templates/index.html
   - Model paths in Voice-auto/

### System Requirements

- **Minimum**:
  - CPU: Dual-core 2.0 GHz
  - RAM: 4GB
  - Camera: 720p 30fps
  - Microphone: Basic built-in
  - Storage: 500MB
  - Python 3.8 or newer

- **Recommended**:
  - CPU: Quad-core 2.5 GHz
  - RAM: 8GB
  - Camera: 1080p 60fps
  - Microphone: External with noise cancellation
  - Storage: 1GB for models and cached data
  - Python 3.10 or newer

## ⚠️ Troubleshooting

### Common Issues

#### Camera Not Working
1. **Access denied**: Check camera permissions in your OS settings
2. **Already in use**: Close other applications that might be using the camera
3. **Device not found**: Ensure webcam is properly connected and recognized by OS
4. **Driver issues**: Update camera drivers

#### Mouse Controller Problems
1. **Missing dependencies**: Run `pip install autopy pyautogui` manually
2. **Controller not starting**: Check console logs for specific errors
3. **Permission issues**: Run the application with appropriate permissions
4. **Platform compatibility**: Use the appropriate commands for your platform:
   - Windows: No additional steps needed
   - macOS: Grant accessibility permissions in System Preferences
   - Linux: Run `xhost +local:` before starting the application

#### Voice Recognition Not Working
1. **Missing Vosk model**: Download the model manually from https://alphacephei.com/vosk/models
2. **Microphone not detected**: Check microphone permissions and settings
3. **Audio input issues**: Select the correct input device in your OS settings

### Platform-Specific Issues

#### Windows
- If you encounter DLL loading errors, install the Visual C++ Redistributable
- For Python import errors, ensure you're using the virtual environment
- If using an older Windows version, run in compatibility mode

#### Linux
- For X11 errors: `export DISPLAY=:0` before running
- For camera issues: `ls -la /dev/video*` to verify camera device
- For permission errors: `chmod +x *.py` to make scripts executable

#### macOS
- For accessibility errors: Re-grant permissions in System Preferences
- For homebrew dependency issues: `brew doctor` to check system status
- For Python path issues: Use `python3` explicitly instead of `python`

### Getting Help

If you're still experiencing issues:
1. Check the console/terminal output for specific error messages
2. Open an issue on our [GitHub Issues page](https://github.com/vigneshbs33/CODEKRAFTERS-Sparsh-Mukthi/issues) with:
   - Your OS and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce the issue
  - Microphone: Noise-canceling
  - Storage: 1GB
  - VR Ready GPU (for VR features)

## ⚙️ Configuration

### Education & Healthcare (edu-hcare.py)
```python
CAM_WIDTH, CAM_HEIGHT = 640, 480  # Camera resolution
FRAME_REDUCTION = 100             # Interaction zone
SMOOTHENING = 7                   # Motion smoothing
CLICK_THRESHOLD = 40              # Gesture detection sensitivity
```

### VR & Gaming (air-contol.py)
```python
ZOOM_THRESHOLD = 10    # VR zoom sensitivity
SCROLL_FACTOR = 20     # Movement multiplier
GRAB_THRESHOLD = 30    # Object grab detection
```

### Voice Control (Voice-auto/)
```python
# Command Mode (final-model.py)
SAMPLE_RATE = 16000    # Audio sampling rate
DURATION = 2           # Listen duration (seconds)
COOLDOWN = 1.5        # Command cooldown time
MODEL_PATH = "vosk-model-small-en-us-0.15"  # Voice model

# Transcription Mode (live_transcriber.py)
BLOCK_SIZE = 2048     # Lower latency for real-time
SHOW_PARTIAL = True   # Show partial results
WORDS_MODE = False    # Optimize for continuous speech
```

### Custom AI Gestures (app.py)
- Training frames: 30
- Detection confidence: 0.7
- Tracking confidence: 0.5

## 🔍 Troubleshooting

1. **Education/Healthcare Mode**:
   - Ensure proper lighting for sterile environments
   - Calibrate for specific room setup
   - Adjust sensitivity for medical gloves

2. **VR/Gaming Mode**:
   - Check VR device compatibility
   - Calibrate room-scale tracking
   - Optimize for game-specific gestures

3. **Custom AI Gestures**:
   - Improve training data quality
   - Adjust detection thresholds
   - Fine-tune gesture mappings

4. **Voice Integration**:
   - Check microphone permissions and settings
   - Calibrate ambient noise levels
   - Test in both command and transcription modes
   - Verify language settings
   - Update voice command dictionary
   - Adjust block size for latency vs accuracy
   - Fine-tune cooldown times for commands

## 🤝 Contributing

We welcome contributions! Areas of interest:

1. **Education**:
   - New learning interaction modes
   - Subject-specific gestures
   - Accessibility features

2. **Healthcare**:
   - Medical device integration
   - Sterile control patterns
   - Patient interaction modes

3. **VR/Gaming**:
   - Game-specific controls
   - VR environment integration
   - Performance optimization

## 🙏 Acknowledgments

- Healthcare professionals for sterile protocol guidance
- Educators for learning interface feedback
- VR developers for integration support
- Open source community

---

<div align="center">
  Made with ❤️ by CODEKRAFTERS<br>
  © 2025 Sparsh Mukthi. All rights reserved.
</div>