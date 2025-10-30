#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sparsh Mukthi - System Compatibility Check Script
This script verifies if your system meets all requirements to run Sparsh Mukthi.
"""

import os
import sys
import platform
import importlib.util
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup_check")

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m' if platform.system() != 'Windows' else ''
    YELLOW = '\033[93m' if platform.system() != 'Windows' else ''
    RED = '\033[91m' if platform.system() != 'Windows' else ''
    BOLD = '\033[1m' if platform.system() != 'Windows' else ''
    RESET = '\033[0m' if platform.system() != 'Windows' else ''

def check_python_version():
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info
    
    logger.info(f"Checking Python version: {sys.version}")
    if current_version >= min_version:
        print(f"{Colors.GREEN}✓ Python {current_version.major}.{current_version.minor}.{current_version.micro} detected{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}✗ Python version {current_version.major}.{current_version.minor}.{current_version.micro} is not supported{Colors.RESET}")
        print(f"  Sparsh Mukthi requires Python {min_version[0]}.{min_version[1]} or higher")
        return False

def check_package(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown version')
            print(f"{Colors.GREEN}✓ {package_name} ({version}){Colors.RESET}")
            return True
        except ImportError:
            print(f"{Colors.YELLOW}! {package_name} found but cannot be imported{Colors.RESET}")
            return False
    else:
        print(f"{Colors.RED}✗ {package_name} not found{Colors.RESET}")
        return False

def check_camera():
    """Check if a camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print(f"{Colors.RED}✗ No camera detected{Colors.RESET}")
            return False
        else:
            ret, frame = cap.read()
            if ret:
                print(f"{Colors.GREEN}✓ Camera is accessible{Colors.RESET}")
                cap.release()
                return True
            else:
                print(f"{Colors.RED}✗ Camera is detected but cannot capture frames{Colors.RESET}")
                cap.release()
                return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error checking camera: {str(e)}{Colors.RESET}")
        return False

def check_microphone():
    """Check if a microphone is available"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print(f"{Colors.GREEN}✓ Microphone is accessible{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}✗ No microphone detected{Colors.RESET}")
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error checking microphone: {str(e)}{Colors.RESET}")
        return False

def check_vosk_model():
    """Check if the Vosk speech recognition model is available"""
    vosk_model_path = os.path.join("Voice-auto", "vosk-model-small-en-us-0.15")
    
    if os.path.exists(vosk_model_path) and os.path.isdir(vosk_model_path):
        print(f"{Colors.GREEN}✓ Vosk speech recognition model found{Colors.RESET}")
        return True
    else:
        print(f"{Colors.YELLOW}! Vosk speech recognition model not found at {vosk_model_path}{Colors.RESET}")
        print(f"  Run 'python Voice-auto/download_vosk_model.py' to download the model")
        return False

def check_requirements_file():
    """Check if all packages in requirements.txt are installed"""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"{Colors.RED}✗ requirements.txt not found{Colors.RESET}")
        return False
    
    print(f"{Colors.BOLD}Checking required packages:{Colors.RESET}")
    missing_packages = []
    
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments and platform-specific dependencies
            if not line or line.startswith('#') or line.startswith('pywin32') or line.startswith('python-xlib'):
                continue
                
            # Handle package specifiers like package>=1.0.0
            package_name = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
            if not check_package(package_name):
                missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n{Colors.YELLOW}Missing packages: {', '.join(missing_packages)}{Colors.RESET}")
        print(f"Run 'pip install -r requirements.txt' to install missing packages")
        return False
    return True

def check_system_dependencies():
    """Check for required system dependencies based on the platform"""
    system = platform.system()
    print(f"{Colors.BOLD}Checking system dependencies for {system}:{Colors.RESET}")
    
    if system == "Windows":
        # Check for Visual C++ Redistributable
        try:
            # This is a simple check that a common DLL from VC++ is available
            import ctypes
            ctypes.cdll.LoadLibrary("vcruntime140.dll")
            print(f"{Colors.GREEN}✓ Microsoft Visual C++ Redistributable is installed{Colors.RESET}")
            return True
        except Exception:
            print(f"{Colors.YELLOW}! Microsoft Visual C++ Redistributable might not be installed{Colors.RESET}")
            print("  This is required for some Python packages. Download from Microsoft's website.")
            return False
            
    elif system == "Linux":
        dependencies = ["ffmpeg", "libportaudio2", "libx11-6", "libxtst6"]
        missing_deps = []
        
        for dep in dependencies:
            try:
                proc = subprocess.run(["dpkg", "-s", dep], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    missing_deps.append(dep)
                    print(f"{Colors.RED}✗ {dep} not installed{Colors.RESET}")
                else:
                    print(f"{Colors.GREEN}✓ {dep} is installed{Colors.RESET}")
            except Exception:
                print(f"{Colors.YELLOW}! Could not check if {dep} is installed{Colors.RESET}")
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"\n{Colors.YELLOW}Missing system dependencies: {', '.join(missing_deps)}{Colors.RESET}")
            print("  Install them with: sudo apt-get install " + " ".join(missing_deps))
            return False
        return True
        
    elif system == "Darwin":  # macOS
        dependencies = ["portaudio", "ffmpeg"]
        missing_deps = []
        
        for dep in dependencies:
            try:
                proc = subprocess.run(["brew", "list", dep], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    missing_deps.append(dep)
                    print(f"{Colors.RED}✗ {dep} not installed{Colors.RESET}")
                else:
                    print(f"{Colors.GREEN}✓ {dep} is installed{Colors.RESET}")
            except Exception:
                print(f"{Colors.YELLOW}! Could not check if {dep} is installed. Is Homebrew installed?{Colors.RESET}")
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"\n{Colors.YELLOW}Missing system dependencies: {', '.join(missing_deps)}{Colors.RESET}")
            print("  Install them with: brew install " + " ".join(missing_deps))
            return False
        return True
    
    else:
        print(f"{Colors.YELLOW}! Unsupported operating system: {system}{Colors.RESET}")
        print("  Sparsh Mukthi is designed for Windows, Linux, and macOS")
        return False

def create_necessary_directories():
    """Create any necessary directories that might be missing"""
    directories = [
        "gesture_output",
        os.path.join("Voice-auto", "recordings")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            print(f"{Colors.GREEN}✓ Directory {directory} exists{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ Failed to create directory {directory}{Colors.RESET}")
            return False
    return True

def main():
    """Run all checks and provide a summary"""
    print(f"{Colors.BOLD}Sparsh Mukthi - System Compatibility Check{Colors.RESET}")
    print(f"{Colors.BOLD}======================================{Colors.RESET}\n")
    
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_requirements_file(),
        "System Dependencies": check_system_dependencies(),
        "Camera": check_camera(),
        "Microphone": check_microphone(),
        "Vosk Model": check_vosk_model(),
        "Directories": create_necessary_directories()
    }
    
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    all_passed = True
    for check, result in checks.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"{check}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! Your system is ready to run Sparsh Mukthi.{Colors.RESET}")
        print(f"Start the application with: python app.py")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Some checks failed. Please address the issues above before running Sparsh Mukthi.{Colors.RESET}")
        print("For more information, see the README.md file or open an issue on GitHub.")
    
    return all_passed

if __name__ == "__main__":
    main()
