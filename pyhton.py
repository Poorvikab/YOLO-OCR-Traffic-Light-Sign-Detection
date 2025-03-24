import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab
from ultralytics import YOLO
import pytesseract  # OCR
import time
import os
from collections import deque

# Set Tesseract Path (Change this based on your system installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\poorv\Documents\Tesseract\tesseract.exe"

# Load YOLOv8 model
model = YOLO("C:/Users/poorv/Desktop/YOLO OCR/runs/detect/train/weights/best.pt")

# Traffic sign categories mapping
category_images = {
    "Amber Light": "images/amber_light.jpg",
    "Green Light": "images/green_light.png",
    "Red Light": "images/red_light.png",
    "Speed Limit 30": "images/speed_30.png",
    "Speed Limit 60": "images/speed_60.png",
    "Speed Limit 80": "images/speed_80.png",
}

# Create folder for saving GUI snapshots
os.makedirs("saved_gui", exist_ok=True)

# Initialize Tkinter window
root = tk.Tk()
root.title("Traffic Sign Detector")

# Set GUI size to full screen height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# Load road image
road_image_path = "images/road.png"
road_img = None
if os.path.exists(road_image_path):
    road_img = Image.open(road_image_path).resize((20, 40))
    road_img = ImageTk.PhotoImage(road_img)

# Create labels for video feed and detections
video_label = tk.Label(root)
video_label.pack(side="left", padx=10, pady=10)

# Create a Frame to hold the scrollbar
scroll_frame = tk.Frame(root)
scroll_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

# Create a Canvas for scrolling
canvas = tk.Canvas(scroll_frame, bg="white", width=300)
canvas.pack(side="left", fill="both", expand=True)

# Add Scrollbar
scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

# Create a Frame inside the Canvas
detection_frame = tk.Frame(canvas, bg="white")
canvas.create_window((0, 0), window=detection_frame, anchor="nw")

# Configure scrolling
canvas.configure(yscrollcommand=scrollbar.set)
detection_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Detection label
detection_label = tk.Label(detection_frame, text="Detected Signs", font=("Arial", 16, "bold"), fg="black", bg="white")
detection_label.pack()

# Stack to store detected signs
detected_signs = deque(maxlen=50)

# Track timestamps of detections to avoid duplicate issues
last_seen = {}

# Open webcam
cap = cv2.VideoCapture("C:/Users/poorv/Desktop/4 video.mp4")

def perform_ocr(cropped_img):
    """Extracts speed limit number from a sign using OCR."""
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    extracted_text = pytesseract.image_to_string(thresh, config="--psm 6")
    extracted_digits = "".join(filter(str.isdigit, extracted_text))
    
    return extracted_digits  # Returns only numbers from OCR result

def save_gui_frame():
    """Capture and save the GUI as an image."""
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    img = ImageGrab.grab((x, y, x + w, y + h))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img.save(f"saved_gui/gui_frame_{timestamp}.png")

def update_gui():
    """Capture frames, run detection, apply OCR, and update GUI."""
    global detected_signs

    success, frame = cap.read()
    if not success:
        root.after(10, update_gui)
        return

    results = model(frame, conf=0.3)  # Run YOLO detection

    new_signs = []
    current_time = time.time()  # Get current timestamp

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            
            # If speed limit sign, apply OCR
            if "Speed Limit" in cls:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                cropped_sign = frame[y1:y2, x1:x2]  # Crop detected sign
                
                ocr_result = perform_ocr(cropped_sign)  # Apply OCR
                print(f"YOLO: {cls}, OCR: {ocr_result}")  # Debugging Output
                
                if ocr_result in ["30", "60", "80"]:  # If OCR detects a valid speed limit
                    cls = f"Speed Limit {ocr_result}"  # Update label based on OCR
                    img_path = category_images.get(cls, None)  # Get corresponding image
                    
                    if img_path and (cls not in last_seen or current_time - last_seen[cls] > 5):
                        new_signs.append(cls)
                        last_seen[cls] = current_time  # Update last seen time
            else:
                if cls not in last_seen or current_time - last_seen[cls] > 5:
                    new_signs.append(cls)
                    last_seen[cls] = current_time  # Update last seen time

    # Add new detections to stack
    for sign in new_signs:
        detected_signs.append(sign)

    # Draw detections
    annotated_frame = results[0].plot()

    # Convert frame for Tkinter
    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((500, 400))
    imgtk = ImageTk.PhotoImage(image=img)

    # Update video feed
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update detected signs in GUI
    for widget in detection_frame.winfo_children():
        if isinstance(widget, tk.Label) and widget != detection_label:
            widget.destroy()

    for sign in reversed(detected_signs):
        img_path = category_images.get(sign, None)
        if img_path:
            img = Image.open(img_path).resize((70, 70))
            img = ImageTk.PhotoImage(img)

            text_label = tk.Label(detection_frame, text=sign, font=("Arial", 14, "bold"), fg="black", bg="white")
            text_label.pack(pady=(5, 0))

            sign_label = tk.Label(detection_frame, image=img, compound="top", bg="white")
            sign_label.photo = img
            sign_label.pack(pady=(0, 10))

            if road_img:
                road_label = tk.Label(detection_frame, image=road_img, bg="white")
                road_label.image = road_img
                road_label.pack(pady=(0, 10))

    detection_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    root.after(50, update_gui)

# Start updating the GUI
update_gui()

# Run Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
