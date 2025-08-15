import cv2

from utils.detector_utils import load_model, connect_to_video, process_boxes, print_vdb_info
from . import config
from .reid import ReID


def main():
    print("Starting program...")
    model = load_model(config.MODEL_PATH)
    re_id = ReID()
    
    cap = connect_to_video(config.STREAM_URL)
    print("Video capture started.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from video stream.")
            break

        result = model.track(frame, tracker=config.TRACKER)[0]

        process_boxes(frame, result, model.names, re_id, config.SKIP_CLASSES)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Unique Object Tracking app", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print_vdb_info(re_id)

if __name__ == "__main__":
    main()