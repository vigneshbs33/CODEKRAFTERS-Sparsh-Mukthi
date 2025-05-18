from flask import Flask, render_template, jsonify, request
import subprocess
import os
import sys
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the workspace root directory and script paths
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(WORKSPACE_ROOT, "gesture_data", "adaptive-gesture-ai")
COLLECT_SCRIPT = os.path.join(SCRIPTS_DIR, "collect_gestures.py")
PREDICT_SCRIPT = os.path.join(SCRIPTS_DIR, "predict_live.py")
GESTURES_FILE = os.path.join(SCRIPTS_DIR, "custom_gestures.json")
MAPPINGS_FILE = os.path.join(SCRIPTS_DIR, "key_mappings.json")

# Default commands and their descriptions
DEFAULT_COMMANDS = {
    "stop": {"function": "Stop current operation", "trained": False, "default": True},
    "left": {"function": "Navigate left (default: LEFT arrow key)", "trained": False, "default": True, "key": "LEFT"},
    "right": {"function": "Navigate right (default: RIGHT arrow key)", "trained": False, "default": True, "key": "RIGHT"},
    "up": {"function": "Navigate up (default: UP arrow key)", "trained": False, "default": True, "key": "UP"},
    "down": {"function": "Navigate down (default: DOWN arrow key)", "trained": False, "default": True, "key": "DOWN"},
    "none": {"function": "No action", "trained": False, "default": True},
    "undo": {"function": "Undo last action (default: CTRL+Z)", "trained": False, "default": True, "key": "CTRL+Z"},
    "redo": {"function": "Redo last action (default: CTRL+Y)", "trained": False, "default": True, "key": "CTRL+Y"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_gestures')
def get_gestures():
    try:
        gestures = DEFAULT_COMMANDS.copy()
        if os.path.exists(GESTURES_FILE):
            with open(GESTURES_FILE, 'r') as f:
                custom_gestures = json.load(f)
                gestures.update(custom_gestures)
        
        # Load key mappings if they exist
        key_mappings = {}
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, 'r') as f:
                key_mappings = json.load(f)
        
        # Update gestures with custom key mappings
        for gesture_name, mapping in key_mappings.items():
            if gesture_name in gestures:
                gestures[gesture_name]['key'] = mapping
        
        return jsonify({"status": "success", "gestures": gestures})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/update_mapping', methods=['POST'])
def update_mapping():
    try:
        data = request.json
        gesture_name = data.get('gesture')
        new_key = data.get('key')
        
        if not gesture_name or not new_key:
            return jsonify({
                "status": "error",
                "message": "Gesture name and key mapping are required"
            })
        
        # Load existing mappings
        mappings = {}
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, 'r') as f:
                mappings = json.load(f)
        
        # Update mapping
        mappings[gesture_name] = new_key
        
        # Save mappings
        with open(MAPPINGS_FILE, 'w') as f:
            json.dump(mappings, f, indent=4)
        
        return jsonify({
            "status": "success",
            "message": f"Updated key mapping for {gesture_name} to {new_key}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        gesture_data = request.json
        gesture_name = gesture_data.get('name', '').strip()
        gesture_function = gesture_data.get('function', '').strip()
        
        if not gesture_name or not gesture_function:
            return jsonify({
                "status": "error",
                "message": "Gesture name and function are required"
            })

        # Save gesture details to JSON file
        gestures = {}
        if os.path.exists(GESTURES_FILE):
            with open(GESTURES_FILE, 'r') as f:
                gestures = json.load(f)
        
        gestures[gesture_name] = {
            "function": gesture_function,
            "trained": False,
            "default": False
        }
        
        with open(GESTURES_FILE, 'w') as f:
            json.dump(gestures, f, indent=4)

        # Run collect_gestures.py with the gesture name
        process = subprocess.Popen(
            [sys.executable, COLLECT_SCRIPT, "--gesture", gesture_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=SCRIPTS_DIR
        )
        logger.info(f"Started training process for gesture: {gesture_name}")
        return jsonify({
            "status": "success",
            "message": f"Training process started for gesture: {gesture_name}"
        })
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_prediction')
def start_prediction():
    try:
        process = subprocess.Popen(
            [sys.executable, PREDICT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=SCRIPTS_DIR
        )
        logger.info("Started prediction process")
        return jsonify({
            "status": "success",
            "message": "Prediction process started"
        })
    except Exception as e:
        logger.error(f"Error starting prediction: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    logger.info("Starting Sparsh Mukthi - Gesture Control Interface")
    logger.info("Access the application at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000) 