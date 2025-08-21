import cv2
import os

import app.config as config

from .detector_utils import process_boxes
from app.config import SessionStatus

def clear_session(session_id, sessions, b_remove_session=False):
    if session_id not in sessions:
        return

    session = sessions[session_id]
    
    if session["status"] == SessionStatus.PENDING:
        return
    
    if session["cap"] and session["cap"].isOpened():
        session["cap"].release()

    if os.path.exists(session["source"]) and config.VIDEO_FOLDER in session["source"]:
        os.remove(session["source"])

    session_path = os.path.join(config.SESSION_FOLDER, f"{session_id}.json")
    if b_remove_session and os.path.exists(session_path):
        os.remove(session_path)
        del sessions[session_id]

def process_video(cap, model, re_id, tracker, skip_classes, start_frame_idx=0, max_frames=-1, resize_shape=None):
    """Process cv2 video frames for object detection and tracking."""
    
    results_all = []
    frame_idxs = start_frame_idx

    while cap.isOpened() and (max_frames < 0 or frame_idxs - start_frame_idx < max_frames):
        success, frame = cap.read()
        if not success:
            break

        if resize_shape and all(resize_shape):
            frame = cv2.resize(frame, (int(resize_shape[0]), int(resize_shape[1])))

        output = model.track(frame, tracker=tracker)[0]
        
        result = process_boxes(frame, frame_idxs, output, model.names, re_id, skip_classes)

        results_all.append(result)
        frame_idxs += 1

    return results_all, frame_idxs-1
