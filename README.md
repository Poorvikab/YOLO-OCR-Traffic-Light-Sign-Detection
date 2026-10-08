# Traffic Sign and Light Detection with OCR

This project uses a trained YOLOv8 model to detect traffic lights and signs in a live webcam feed or a video. For speed-limit signs, it runs Tesseract OCR on the detected sign, accepts the configured speed values, and displays the latest valid result. It saves an annotated video and a CSV detection log locally.

> This is a computer-vision demonstration, not a certified road-safety system. Do not use its output to control a vehicle or make safety-critical decisions.

## What is included on GitHub

- The Python inference script, dependency list, YOLO dataset configuration, and this guide.
- `runs/detect/train/weights/best.pt`, the trained checkpoint used by the app.
- `images/` example images and dataset attribution/metadata files.

The datasets, downloaded base-model weights, raw/sample videos, generated training results, and local detection outputs are excluded to keep the repository manageable. The local `data.yaml` refers to dataset folders that are intentionally not committed, so training requires downloading the attributed datasets and restoring the expected folder layout. Inference with the included checkpoint does not require the training dataset.

## Requirements

- Windows, macOS, or Linux with Python 3.9 or newer.
- A webcam for live inference, or a local video file.
- Tesseract OCR installed separately from Python. Install it from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), and make sure its executable is on `PATH`. If it is not, pass its full path with `--tesseract-cmd`.
- A CPU can run inference; a compatible NVIDIA GPU and CUDA-enabled PyTorch may improve speed. Install the PyTorch build appropriate for your system if the default installation does not provide the desired GPU support.

## Setup

1. Clone the repository and open a terminal in its root folder:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the Python dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   For GPU inference, first install the PyTorch/torchvision build matching your CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt` if needed.

4. Connect a webcam, or prepare a video file. Confirm that Tesseract is installed and available:

   ```bash
   tesseract --version
   ```

## Run inference

From the repository root, start the webcam (the default source):

```bash
python detect.py
```

Use a video file instead:

```bash
python detect.py --source path/to/video.mp4
```

Choose another model, output directory, confidence threshold, or Tesseract executable:

```bash
python detect.py --weights runs/detect/train/weights/best.pt --source path/to/video.mp4 --output-dir saved_results --conf 0.3 --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Press **q** in the preview window to stop. The output directory is relative to the current terminal directory unless an absolute path is supplied. Each run creates or overwrites:

- `saved_results/prediction.mp4` — annotated video.
- `saved_results/ObjectDetection_Logs.csv` — timestamp, detected class, YOLO confidence, estimated per-detection FPS, and OCR text.

These results stay on the machine running the app and are ignored by Git. The live preview requires a desktop display. A video file is processed until it ends or **q** is pressed.

## Processing flow

1. Load the trained YOLO checkpoint from `runs/detect/train/weights/best.pt`.
2. Open the selected webcam or video and read frames.
3. Run YOLO detection at the selected confidence threshold (default `0.3`).
4. For detections whose class name contains `Speed Limit`, crop the box and use grayscale thresholding followed by Tesseract (`--psm 6`) to read digits. Only `5`, `10`, `15`, `20`, `25`, `30`, `60`, and `80` are accepted; invalid OCR results are skipped.
5. Draw accepted detections and the most recently accepted speed limit on the preview and output video; append detections to the CSV log.
6. Release the camera/video and output files when the input ends or the user quits.

## Model performance

The local training log at `runs/detect/train/results.csv` reports these final (10th epoch) **validation** metrics:

| Metric | Value |
|---|---:|
| Precision | 0.686 |
| Recall | 0.739 |
| mAP@0.50 | 0.746 |
| mAP@0.50:0.95 | 0.515 |

These figures are transcribed from the local training run; generated training logs are excluded from Git. They are object-detection metrics on that run's validation split, not a single overall “accuracy” score and not a guarantee of real-world performance. They do not measure OCR character/speed-limit accuracy. Results vary with lighting, distance, blur, camera angle, hardware, and input video. The repository does not include an independent test-set evaluation or a separate OCR accuracy benchmark.

## Training and data

`data.yaml` defines the train, validation, and test image directories and class names. The training datasets themselves are not included. The root `README.dataset.txt` and the attribution links below identify the sources; the datasets' original metadata files are not included. The sources identify their datasets as CC BY 4.0. Download the datasets from the credited Roboflow pages, review their terms and attribution requirements, and arrange their image/label folders to match `data.yaml` before training.

The local run configuration used `yolov8n.pt`, 640-pixel images, batch size 16, and 10 epochs. Training outputs and their machine-specific configuration are excluded from Git; this summary is provided as context, not as a portable training command. The inference app uses the trained `best.pt`, not the downloaded `yolov8n.pt`.

## Troubleshooting

- **Model file not found:** run the command from the repository root and verify `runs/detect/train/weights/best.pt` exists.
- **Tesseract not found:** install Tesseract and add it to `PATH`, or pass its executable with `--tesseract-cmd`.
- **Webcam cannot open:** close other webcam apps, check OS camera permissions, or use `--source path/to/video.mp4`.
- **No preview window:** run in a desktop session; this script uses OpenCV's GUI window.
- **CUDA unavailable:** inference can use CPU. GPU acceleration depends on the installed PyTorch build and compatible drivers/CUDA.

## Dataset attribution

- [Traffic sign and light dataset](https://universe.roboflow.com/poorvika/traffic_sign_and_light_detection) — source metadata identifies CC BY 4.0.
- [OCR dataset](https://universe.roboflow.com/poorvika/ocr-ubdj0) — source metadata identifies CC BY 4.0.
- The pre-trained YOLOv8 base weights are not included. See [Ultralytics licensing](https://www.ultralytics.com/license) and the terms of any weights you download.
