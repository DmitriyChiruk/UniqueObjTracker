import cv2

def process_boxes(frame, frame_idx, result, names, re_id, skip_classes=set()):
    """
    Process the detected boxes and draw them on the frame.

    :param frame: The video frame to process.
    :param frame_idx: The index of the frame being processed.
    :param result: The detection result containing bounding boxes and class information.
    :param names: The class labels for the detected objects.
    :param re_id: The ReID model for re-identification.

    :param skip_classes: A set of class IDs to skip during processing.
    """
    if result.boxes is None or result.boxes.xyxy is None:
        return {"frame_index": frame_idx, "detections": []}
    
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    clases = result.boxes.cls.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
        
    ids = result.boxes.id.detach().cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
    
    detections = []
    for (tlx, tly, brx, bry), cls, conf in zip(boxes, clases, confs):
        if skip_classes and cls in skip_classes:
            continue
        
        tlx, tly, brx, bry = map(int, (tlx, tly, brx, bry))
        
        id = None
        label = names[int(cls)]
        
        if re_id:
            crop = frame[tly:bry, tlx:brx]
            id = re_id.add(crop, metadata={"label": label})[:8]
        
        detections.append({
            "id": id,
            "label": label,
            "cls": int(cls),
            "conf": float(conf),
            "bbox": [tlx, tly, brx, bry],
        })

    return {"frame_index": frame_idx, "detections": detections}

def process_video(cap, model, re_id, tracker, skip_classes, max_frames=-1, resize_shape=None):
    """Process cv2 video frames for object detection and tracking."""
    
    results_all = []
    frame_idxs = 0

    while cap.isOpened() and (max_frames < 0 or frame_idxs < max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if resize_shape and all(resize_shape):
            frame = cv2.resize(frame, (int(resize_shape[0]), int(resize_shape[1])))

        output = model.track(frame, tracker=tracker)[0]
        
        result = process_boxes(frame, frame_idxs, output, model.names, re_id, skip_classes)
        results_all.append(result)
        frame_idxs += 1

    return results_all