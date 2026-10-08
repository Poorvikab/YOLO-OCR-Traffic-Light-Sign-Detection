# Traffic Sign and Light Detection with OCR

This project uses a trained YOLOv8 model to detect traffic lights and signs in a live webcam feed or a video. For speed-limit signs, it runs Tesseract OCR on the detected sign, accepts the configured speed values, and displays the latest valid result. It saves an annotated video and a CSV detection log locally.

> This is a computer-vision demonstration, not a certified road-safety system. Do not use its output to control a vehicle or make safety-critical decisions.

## What is included on GitHub

- The Python inference script, dependency list, YOLO dataset configuration, and this guide.
- `runs/detect/train/weights/best.pt`, the trained checkpoint used by the app.
- `images/` example images and dataset attribution/metadata files.

The datasets, downloaded base-model weights, raw/sample videos, generated training results, and local detection outputs are excluded to keep the repository manageable. The local `data.yaml` refers to dataset folders that are intentionally not committed, so training requires downloading the attributed datasets and restoring the expected folder layout. Inference with the included checkpoint does not require the training dataset.

## Trained Model

The trained YOLOv8 model is available on Hugging Face:

[YOLOv8 Traffic Sign & Light Detection Model](https://huggingface.co/poorvika4566/YOLOv8-Traffic-Sign-Light-OCR)

The model weights are available as `best.pt`.

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
