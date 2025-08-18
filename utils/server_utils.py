import cv2

from .detector_utils import process_boxes

def process_video(cap, model, re_id, tracker, skip_classes, max_frames=-1, resize_shape=None):
    """Process cv2 video frames for object detection and tracking."""
    
    results_all = []
    frame_idxs = 0

    while cap.isOpened() and (max_frames < 0 or frame_idxs < max_frames):
        success, frame = cap.read()
        if not success:
            break

        if resize_shape and all(resize_shape):
            frame = cv2.resize(frame, (int(resize_shape[0]), int(resize_shape[1])))

        output = model.track(frame, tracker=tracker)[0]
        
        result = process_boxes(frame, frame_idxs, output, model.names, re_id, skip_classes)

        results_all.append(result)
        frame_idxs += 1

    return results_all