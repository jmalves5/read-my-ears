# config.py
from pathlib import Path

# Base path
WORKSPACE_PATH = Path(__file__).parent.parent

# raw data directory
DATASET_DIR = Path(f"{WORKSPACE_PATH}/ReadMyEars_Dataset/data")

# output directory
OUTPUT_DIR = Path("./output")

# model directory
EAR_DETECTION_MODEL_PATH = Path(f"{WORKSPACE_PATH}/horse-face-ear-detection/horse_ear_detection/yolov8n_horse_ear_detection.pt")
FACE_DETECTION_MODEL_PATH = Path(f"{WORKSPACE_PATH}/horse-face-ear-detection/horse_face_detection/yolov8n_horse_face_detection.pt")
