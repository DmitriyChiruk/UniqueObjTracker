import cv2

from utils.detector_utils import load_model, connect_to_video, process_boxes, print_vdb_info, process_video
from . import config
from .reid import ReID


def main():
    print("Starting program...")
    model = load_model(config.MODEL_PATH)
    re_id = ReID()
    
    cap = connect_to_video(config.STREAM_URL)
    print("Video capture started.")

    process_video(cap, model, re_id, config.TRACKER, config.SKIP_CLASSES)

    print_vdb_info(re_id)

if __name__ == "__main__":
    main()