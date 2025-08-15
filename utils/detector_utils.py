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

def draw_boxes(frame, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on the frame."""
    height, width = frame.shape[:2]
    for i, detection in enumerate(detections):
        
        tlx, tly, brx, bry = detection["bbox"]
        tlx, tly, brx, bry = map(int, (tlx * width, tly * height, brx * width, bry * height))
        text = f"id: {detection['id'][:8]} {detection['label']} {detection['conf']:.2f}"

        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, thickness)
        cv2.putText(frame, text, (tlx, max(20, tly-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def process_boxes(frame, frame_idx, result, labels, re_id, skip_classes=set(), color=(0, 255, 0), thickness=2):
    """
    Process the detected boxes.

    :param frame: The video frame to process.
    :param result: The detection result containing bounding boxes and class information.
    :param labels: The class labels for the detected objects.
    :param re_id: The ReID model for re-identification.

    :param classes_threshold: The class ID threshold for filtering detections. If stated -1, no filtering is applied.
    :param color: The color to use for drawing bounding boxes. Defaults to green.
    """

    if result.boxes is None or result.boxes.xyxy is None:
        return {"frame_index": frame_idx, "detections": []}
    
    height, width = frame.shape[:2]
    boxes = result.boxes.xyxyn.detach().cpu().numpy()
    clases = result.boxes.cls.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
        
    ids = result.boxes.id.detach().cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
    detections = []
    ids_pool = {}
    
    for (tlxn, tlyn, brxn, bryn), cls, conf, id in zip(boxes, clases, confs, ids):
        if skip_classes and cls in skip_classes:
            continue

        tlx, tly, brx, bry = map(int, (tlxn * width, tlyn * height, brxn * width, bryn * height))

        id = int(id) if id is not None else None
        label = labels[int(cls)]
        
        if re_id:
            crop = frame[tly:bry, tlx:brx]
            # rid, vdb_embed = re_id.add(crop, metadata={"label": label})
            
            cur_embed = re_id.get_embedding(crop)
            rid = re_id.search(cur_embed)
            if not rid:
                rid = re_id.append(cur_embed, metadata={"label": label})
                ids_pool[rid] = {
                    "embedding": cur_embed,
                    "label": label,
                    "cls": int(cls),
                    "conf": float(conf),
                    "bbox": [tlxn, tlyn, brxn, bryn],
                }
            else:
                vdb_embed = re_id.search_embedding(rid)
                
                # if id in DB but not in pool
                if rid not in ids_pool:
                    ids_pool[rid] = {
                        "embedding": cur_embed,
                        "label": label,
                        "cls": int(cls),
                        "conf": float(conf),
                        "bbox": [tlxn, tlyn, brxn, bryn],
                    }
                # if id in db and pool => compare which is closer, the other should be added as a new instance to vdb
                else:
                    saved_embed = ids_pool[rid]["embedding"]

                    dist_cur = re_id.calc_distance(cur_embed, vdb_embed)
                    dist_saved = re_id.calc_distance(saved_embed, vdb_embed)
                    
                    if dist_cur < dist_saved:
                        ids_pool[rid]["embedding"] = cur_embed
                        new_rid = re_id.append(saved_embed, metadata={"label": label})
                        new_embed = saved_embed
                    else:
                        ids_pool[rid]["embedding"] = saved_embed
                        new_rid = re_id.append(cur_embed, metadata={"label": label})
                        rid = new_rid
                        new_embed = cur_embed

                    ids_pool[new_rid] = {
                        "embedding": new_embed,
                        "label": label,
                        "cls": int(cls),
                        "conf": float(conf),
                        "bbox": [tlxn, tlyn, brxn, bryn],
                    }

        # detections.append({
        #     "id": rid if re_id else id,
        #     "label": label,
        #     "cls": int(cls),
        #     "conf": float(conf),
        #     "bbox": [tlxn, tlyn, brxn, bryn],
        # })

    for id, data in ids_pool.items():
        detections.append({
            "id": id,
            "label": data["label"],
            "cls": data["cls"],
            "conf": data["conf"],
            "bbox": data["bbox"],
        })

    return {"frame_index": frame_idx, "detections": detections}


def process_video(cap, model, re_id, tracker, skip_classes, resize_shape=(1280, 720)):
    """Process cv2 video frames for object detection and tracking."""
    
    idxs = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from video stream.")
            break

        result = model.track(frame, tracker=tracker)[0]

        res = process_boxes(frame, idxs, result, model.names, re_id, skip_classes)
        idxs += 1

        draw_boxes(frame, res["detections"])
        
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
    assert re_id, "ReID model is not initialized."
    
    print(f"Number of ReID objects: {re_id.count()}")
    for id in re_id.list_ids():
        print(f"ReID object ID: {id}")
