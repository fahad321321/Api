import os
import gdown
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from flask_cors import CORS  # Import CORS to handle cross-origin issues

app = Flask(__name__)

# Enable CORS
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
MODEL_PATH = "RealESRGAN_x4plus.pth"

# Google Drive file ID
file_id = '1Oymd6vySra103Rx28wb6KxsFg7-aYAig'

# Get the current working directory
current_directory = os.getcwd()

# Destination path where you want to save the file in the current directory
destination = os.path.join(current_directory, 'RealESRGAN_x4plus.pth')



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check if the file already exists
if not os.path.exists(destination):
    # Construct the Google Drive download URL
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the file
    gdown.download(url, destination, quiet=False)
else:
    print("File already exists!")

def enhance_image(input_path, output_path):
    try:
        print("[INFO] Starting image enhancement...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        
        print("[INFO] Initializing RealESRGANer...")
        upsampler = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model,
            tile=128,        # ✅ Optimized for CPU
            tile_pad=10,
            pre_pad=0,
            half=False,      # ✅ Use full precision for CPU
            device=device
        )

        print(f"[INFO] Opening input image: {input_path}")
        img = Image.open(input_path).convert("RGB")

        print("[INFO] Running enhancement... Please wait.")
        output, _ = upsampler.enhance(np.array(img), outscale=4)

        print(f"[INFO] Saving output to: {output_path}")
        Image.fromarray(output).save(output_path)

        print("[INFO] Enhancement completed successfully!")
    except Exception as e:
        print(f"[ERROR] Error in enhancement: {e}")
        raise

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        print("[INFO] Received upload request")

        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        output_filename = 'output.jpg'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        print("[INFO] Saving input file...")
        file.save(input_path)

        print("[INFO] Calling enhance_image() function")
        enhance_image(input_path, output_path)

        image_url = f"http://127.0.0.1:5000/output/{output_filename}"
        print(f"[INFO] Returning output URL: {image_url}")

        return jsonify({"status": "success", "enhanced_image_url": image_url})
    
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/output/<filename>')
def get_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
