import argparse
import cv2
import csv
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import shutil
import time

import pytesseract
from ultralytics import YOLO


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)
VALID_SPEED_VALUES = {"5", "10", "15", "20", "25", "30", "60", "80"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect traffic signs/lights and read speed limits from a webcam or video."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="YOLO weights file (default: the trained checkpoint included in this repo).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (default: 0) or path to an input video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved_results"),
        help="Directory for the annotated video and CSV log (default: saved_results).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO confidence threshold from 0 to 1 (default: 0.3).",
    )
    parser.add_argument(
        "--tesseract-cmd",
        help="Tesseract executable path; by default, Tesseract is searched for on PATH.",
    )
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    return args


def configure_tesseract(command):
    if command is None:
        command = shutil.which("tesseract")
        if command is None and os.name == "nt":
            windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if windows_path.is_file():
                command = str(windows_path)
    elif not Path(command).is_file():
        command = shutil.which(command)

    if command is None:
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install Tesseract, add it to PATH, "
            "or pass --tesseract-cmd with its executable path."
        )
    pytesseract.pytesseract.tesseract_cmd = command


def perform_ocr(cropped_img):
    """Extract digits from a detected speed-limit sign."""
    if cropped_img.size == 0:
        return ""
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    extracted_text = pytesseract.image_to_string(thresh, config="--psm 6")
    digits = "".join(filter(str.isdigit, extracted_text))
    return digits


def main():
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO weights file not found: {args.weights}")
    configure_tesseract(args.tesseract_cmd)

    model = YOLO(str(args.weights))
    print(f"[INFO] Loaded model classes: {model.names}")

    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(f"Video source has invalid frame dimensions: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "ObjectDetection_Logs.csv"
    output_path = args.output_dir / "prediction.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path), fourcc, fps, (frame_width, frame_height)
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    speed_limit_queue = deque(maxlen=5)
    print("[INFO] Starting detection... Press 'q' to quit.")
    try:
        with log_path.open(mode="w", newline="", encoding="utf-8") as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(
                ["Timestamp", "Object Class", "Confidence", "FPS", "OCR_Value"]
            )

            while True:
                success, frame = cap.read()
                if not success:
                    print("[INFO] End of video or frame could not be captured.")
                    break

                start_time = time.time()
                results = model(frame, conf=args.conf, verbose=False)
                annotated_frame = frame.copy()

                for result in results:
                    for box in result.boxes:
                        cls_name = model.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        ocr_value = ""

                        if "Speed Limit" in cls_name:
                            cropped = frame[y1:y2, x1:x2]
                            ocr_value = perform_ocr(cropped)
                            if ocr_value not in VALID_SPEED_VALUES:
                                continue
                            cls_name = f"Speed Limit {ocr_value}"
                            queue_timestamp = datetime.now().strftime("%H:%M:%S")
                            speed_limit_queue.append((queue_timestamp, cls_name))

                        cv2.rectangle(
                            annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )
                        cv2.putText(
                            annotated_frame,
                            cls_name,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        frame_fps = 1 / (time.time() - start_time + 1e-6)
                        csv_writer.writerow(
                            [
                                timestamp,
                                cls_name,
                                round(confidence, 2),
                                round(frame_fps, 2),
                                ocr_value,
                            ]
                        )

                if speed_limit_queue:
                    latest_sign = speed_limit_queue[-1]
                    cv2.putText(
                        annotated_frame,
                        f"Latest Detected: {latest_sign[1]}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Time: {latest_sign[0]}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                out.write(annotated_frame)
                cv2.imshow("YOLO Detection + OCR (Press Q to quit)", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Detection finished. Logs and video saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
