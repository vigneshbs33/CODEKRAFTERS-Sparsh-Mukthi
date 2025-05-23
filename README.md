# Sparsh Mukthi - Advanced Gesture Control System

<div align="center">
  <img src="https://i.ibb.co/217zjLmn/temp.webp" alt="Sparsh Mukthi Logo" width="200"/>
  <p><i>Touchless Control System for Education, Healthcare, and VR Gaming</i></p>
</div>

## 🌟 Overview

Sparsh Mukthi is a comprehensive gesture recognition system designed for three key domains: Education, Healthcare, and VR Gaming. Built with modern computer vision technologies, it provides touchless interaction solutions particularly valuable in scenarios requiring hygiene (healthcare), immersive learning (education), and natural VR control (gaming).

### Key Features

- **Education & Healthcare Mode (edu-hcare.py)**:
  - Touchless interaction for sterile environments
  - Immersive learning experiences
  - Virtual anatomy manipulation
  - Presentation control for educators
  - COVID-safe interaction protocols

- **VR & Gaming Control (air-contol.py)**:
  - Natural VR navigation
  - Precise game control gestures
  - 3D space manipulation
  - Advanced gesture combinations
  - Smooth motion tracking

- **Voice Integration**:
  - Natural language commands
  - Voice-gesture hybrid control
  - Multi-language support
  - Custom voice command mapping
  - Noise-resistant recognition
  - Real-time voice feedback

- **Custom AI Gestures (Main Application)**:
  - Train personalized gestures
  - Map to keyboard/mouse actions
  - Real-time recognition
  - Modern cyberpunk interface
  - Extensible command system

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or depth camera
- Git (for cloning)
- Windows/Linux/MacOS
- VR headset (optional, for VR features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vigneshbs33/CODEKRAFTERS-Sparsh-Mukthi
cd CODEKRAFTERS-Sparsh-Mukthi
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Component-Specific Setup

1. **Education & Healthcare Mode**:
```bash
python Main-flow-gesture/edu-hcare.py
```
- Calibrate camera position
- Set sterile interaction zone
- Configure gesture sensitivity

2. **VR & Gaming Control**:
```bash
python Main-flow-gesture/air-contol.py
```
- Connect VR devices (if applicable)
- Adjust tracking sensitivity
- Set game-specific gestures

3. **Voice Control Setup**:
```bash
cd Voice-auto
# For command-based voice control:
python final-model.py
# For real-time transcription:
python live_transcriber.py
```
- Download Vosk model (vosk-model-small-en-us-0.15)
- Configure voice commands in commands.json
- Test microphone input
- Choose between:
  - Command mode: Voice-to-action mapping with feedback
  - Transcription mode: Real-time speech-to-text

4. **Custom AI Gestures**:
```bash
python app.py
```
- Access web interface at http://127.0.0.1:5000
- Train custom gestures
- Configure mappings

## 🛠️ Technical Details

### Architecture

```
sparsh-mukthi/
├── app.py                 # Main AI gesture application
├── requirements.txt       # Project dependencies
├── Main-flow-gesture/    # Specialized control modules
│   ├── edu-hcare.py      # Education/Healthcare interface
│   └── air-contol.py     # VR/Gaming controller
├── Voice-auto/           # Voice control system
│   ├── final-model.py    # Voice recognition engine
│   ├── commands.json     # Voice command mappings
│   └── live_transcriber.py # Real-time transcription
├── static/               # Web assets
├── templates/            # Web interface
├── models/               # AI models
├── gesture_data/         # Training data
└── gesture_output/       # Recognition output
```

### Technologies Used

- **Core**: Python 3.8+, OpenCV 4.8+
- **AI/ML**: MediaPipe, NumPy, Scikit-learn
- **VR Integration**: PyGame, AutoPy
- **Voice Processing**: Vosk, sounddevice, pyttsx3
- **Web Interface**: Flask, SocketIO
- **Input Control**: PyAutoGUI, Pynput

### System Requirements

- **Minimum**:
  - CPU: Dual-core 2.0 GHz
  - RAM: 4GB
  - Camera: 720p 30fps
  - Microphone: Basic built-in
  - Storage: 500MB

- **Recommended**:
  - CPU: Quad-core 2.5 GHz
  - RAM: 8GB
  - Camera: 1080p 60fps
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