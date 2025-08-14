import os
import cv2
import yt_dlp

from ultralytics import YOLO, RTDETR

def load_model(model_path):
    """Load a YOLO or RTDETR model from the specified path."""
    
    assert model_path, "Model path cannot be empty."
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    assert model_path[-3:] == ".pt", f"Model path {model_path} must end with .pt"

    if 'yolo' in model_path:
        return YOLO(model_path)
    elif 'rtdetr' in model_path:
        return RTDETR(model_path)
    else:
        raise ValueError(f"Unknown model type for path {model_path}")

def connect_to_video(stream_url):
    """
    Connect to a video stream from a URL or a local camera.
    """
    
    if "youtube.com" in stream_url:
        ydl_opts = {
            "quiet": True,
            "format": "best",
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(stream_url, download=False)
            direct_url = info["url"]

        print("Direct stream URL:", direct_url)
        return cv2.VideoCapture(direct_url)
    
    return cv2.VideoCapture(stream_url)
    
def process_boxes(frame, result, labels, re_id, skip_classes=set(), color=(0, 255, 0)):
    """
    Process the detected boxes and draw them on the frame.

    :param frame: The video frame to process.
    :param result: The detection result containing bounding boxes and class information.
    :param labels: The class labels for the detected objects.
    :param re_id: The ReID model for re-identification.

    :param classes_threshold: The class ID threshold for filtering detections. If stated -1, no filtering is applied.
    :param color: The color to use for drawing bounding boxes. Defaults to green.
    """
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    clases = result.boxes.cls.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
        
    ids = result.boxes.id.detach().cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
    
    for (tlx, tly, brx, bry), cls, conf, id in zip(boxes, clases, confs, ids):
        if skip_classes and cls in skip_classes:
            continue
        
        tlx, tly, brx, bry = map(int, (tlx, tly, brx, bry))
        id = int(id) if id is not None else None
        label = labels[int(cls)]
        text = f"{label} {conf:.2f}"
        
        if re_id:
            crop = frame[tly:bry, tlx:brx]
            id = re_id.add(crop, metadata={"label": label})[:8]

        if id:
            text = f"id: {id} " + text
        
        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, 2)
        cv2.putText(frame, text, (tlx, max(20, tly-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_video(cap, model, re_id, tracker, skip_classes, resize_shape=(1280, 720)):
    """Process cv2 video frames for object detection and tracking."""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from video stream.")
            break

        result = model.track(frame, tracker)[0]

        process_boxes(frame, result, model.names, re_id, skip_classes)

        frame = cv2.resize(frame, resize_shape)
        cv2.imshow("Unique Object Tracking app", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def print_vdb_info(re_id):
    """
    Print information about the ReID database.
    """
    print(f"Number of ReID objects: {re_id.count()}")
    for id in re_id.list_ids():
        print(f"ReID object ID: {id}")
