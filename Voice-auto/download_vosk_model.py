import os
import sys
import requests
import zipfile
import shutil
import time

# Define the model to download
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)
ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{MODEL_NAME}.zip")

def download_file(url, local_path):
    """Download a file with progress indication"""
    print(f"Downloading {url} to {local_path}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    if total_size == 0:
        print("Warning: Content length is 0, download may fail")
    
    downloaded = 0
    start_time = time.time()
    
    with open(local_path, 'wb') as file:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            file.write(data)
            
            # Calculate progress and speed
            progress = int(50 * downloaded / total_size) if total_size > 0 else 0
            elapsed = time.time() - start_time
            speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0  # MB/s
            
            # Print progress bar
            sys.stdout.write(f"\r[{'#' * progress}{' ' * (50 - progress)}] {downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB ({speed:.1f} MB/s)")
            sys.stdout.flush()
    
    print("\nDownload complete!")

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    # Create a temporary extraction directory
    temp_dir = f"{extract_to}_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the model directory in the extracted files
    extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
    if not extracted_dirs:
        print("Error: No directories found in the extracted zip file")
        return False
    
    # Move the contents to the final location
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to)
    
    # The model might be in a subdirectory with the same name
    source_dir = os.path.join(temp_dir, extracted_dirs[0])
    
    # Copy all files from the source directory to the destination
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(extract_to, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("Extraction complete!")
    return True

def main():
    print(f"=== Downloading Vosk model: {MODEL_NAME} ===")
    
    # Check if model already exists
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model directory {MODEL_DIR} already exists and is not empty.")
        choice = input("Do you want to download and reinstall the model? (y/n): ")
        if choice.lower() != 'y':
            print("Aborted.")
            return
        
    # Download the model
    try:
        download_file(MODEL_URL, ZIP_PATH)
    except Exception as e:
        print(f"Error downloading the model: {e}")
        return
    
    # Extract the model
    try:
        success = extract_zip(ZIP_PATH, MODEL_DIR)
        if success:
            print(f"Model {MODEL_NAME} has been successfully downloaded and installed to {MODEL_DIR}")
            
            # Clean up the zip file
            os.remove(ZIP_PATH)
            print(f"Removed zip file: {ZIP_PATH}")
        else:
            print("Failed to extract the model")
    except Exception as e:
        print(f"Error extracting the model: {e}")

if __name__ == "__main__":
    main()
