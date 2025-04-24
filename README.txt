🛠️ RealESRGAN Flask API Setup Instructions

🔹 Step 1: Create Virtual Environment (optional but recommended)
--------------------------------------------
python -m venv venv
source venv/bin/activate  (Linux/macOS)
venv\Scripts\activate   (Windows)

🔹 Step 2: Install Required Packages
--------------------------------------------
pip install -r requirements.txt

🔹 Step 3: Download Model File
--------------------------------------------
Download this file and place it in the same folder as main.py:
➡️ https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth

🔹 Step 4: Run the API
--------------------------------------------
python main.py

🔹 Step 5: Use the API
--------------------------------------------
POST to: http://localhost:5000/upload
Form-data key: file
Value: (upload your image file)

✅ Response will return image URL like:
http://localhost:5000/output/output.jpg
