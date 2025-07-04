# YOLO-OCR
🚦 YOLO-Based Traffic Light & Sign Detection
A real-time object detection project using YOLOv8 to detect traffic lights and road signs in live camera feeds. The dataset is sourced and preprocessed using Roboflow, and the model is trained for accurate detection in autonomous driving or traffic monitoring use-cases.

📁 Dataset
🗂️ Source: Roboflow

✅ Contains annotated images of:

🟥 Red, 🟨 Yellow, 🟩 Green traffic lights

🚧 Road signs (e.g., Stop, Yield, Speed Limit, etc.)

💾 Format: YOLOv8 (TXT annotations)

📦 Augmented with brightness and rotation variations

🧠 Model & Training

⚙️ Model: YOLOv8

🔧 Framework: Python, PyTorch, Ultralytics YOLOv8

🏋️ Training:

Epochs: 100

Batch Size: 16

Image Size: 640x640

Validation split: 20%

🧪 Achieved: XX% mAP@0.5 on test set
