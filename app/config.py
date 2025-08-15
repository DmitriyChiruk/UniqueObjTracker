import os
from dotenv import load_dotenv

load_dotenv()

STREAM_URL = os.getenv("STREAM_URL", 0)
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
TRACKER = os.getenv("TRACKER", "bytetrack.yaml")

coco_num_classes = int(os.getenv("COCO_NUM_CLASSES", 80))
classes_threshold = int(os.getenv("CLASSES_THRESHOLD", 0))
SKIP_CLASSES = set(range(classes_threshold, coco_num_classes))

CHROMADB_PATH = os.getenv("CHROMADB_PATH", "objects_ids")
REID_MODEL = os.getenv("REID_MODEL", "osnet_x1_0")

input_shape = os.getenv("REID_INPUT_SHAPE")
input_shape = map(int, input_shape.split(","))
REID_INPUT_SHAPE = tuple(input_shape)

REID_SIM_THRESHOLD = float(os.getenv("REID_SIM_THRESHOLD", "0.7"))
REID_DISTANCE = os.getenv("REID_DISTANCE", "cosine")
