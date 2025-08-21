import os
import uuid
import cv2
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Union

import app.config as config

from app.config import SessionStatus
from app.reid import ReID
from utils.detector_utils import load_model, connect_to_video
from utils.server_utils import process_video, clear_session

app = FastAPI(title="Unique Object Tracking App")

REID = ReID()
MODEL = load_model(config.MODEL_PATH)
SESSIONS = {}

def session2response(session_id):
    if session_id not in SESSIONS:
        return JSONResponse(
            {"error": f"Session '{session_id}' not found."},
            status_code=404
        )
    
    session = SESSIONS[session_id]
    
    response = {
        "session_id": session_id,
        "status": session["status"].value,
        "source": session["source"],
        "num_frames": len(session["frames"]),
        "processed_frames": session["frames"]
    }
    
    if not os.path.exists(config.SESSION_FOLDER):
        os.makedirs(config.SESSION_FOLDER)

    response_path = os.path.join(config.SESSION_FOLDER, f"{session_id}.json")
    with open(response_path, "w") as f:
        json.dump(response, f)
        
    return JSONResponse(response, media_type="application/json")

def process_chunk(session_id):
    if session_id not in SESSIONS:
        return
    
    if SESSIONS[session_id]["status"] != SessionStatus.PROCESSING:
        return

    session = SESSIONS[session_id]
    cap = session["cap"]
    max_frames = session["max_frames"]
    
    if session["max_frames"] > 0:
        max_frames = min(session["chunk_sz"], session["max_frames"] - session["next_frame"])
        max_frames = min(max_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT) - session["next_frame"])

    frames, last_frame_idx = process_video(
        cap, 
        MODEL,
        REID,
        config.TRACKER,
        config.SKIP_CLASSES,
        session["next_frame"],
        max_frames,
        session["resize_shape"]
        )
    
    session["frames"].extend(frames)

    session["status"] = SessionStatus.PENDING
    session["next_frame"] = last_frame_idx + 1
    

    b_done_condition = session["next_frame"] >= session['max_frames'] - 1 and session['max_frames'] > 0
    b_done_condition = b_done_condition or session["next_frame"] >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    
    if not cap.isOpened() or b_done_condition:
        session["status"] = SessionStatus.DONE
        cap.release()

    clear_session(session_id, SESSIONS, b_remove_session=False)

    return session2response(session_id)

@app.get("/")
def read_root():
    return {"appname": "Unique Object Tracking App"}

@app.post("/process")
async def process(
    stream_url: Union[str, None] = Form(None),
    video_file: Union[UploadFile, None] = File(None),
    max_frames: int = Form(1000),
    chunk_sz: int = Form(200)
    ):

    session_id = str(uuid.uuid4())
    if video_file:
        video_path = os.path.join(config.VIDEO_FOLDER, f"{session_id}_{video_file.filename}")

        if not os.path.exists(config.VIDEO_FOLDER):
            os.makedirs(config.VIDEO_FOLDER)

        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        source = video_path
        cap = cv2.VideoCapture(source)
    elif stream_url:
        source = stream_url
        cap = connect_to_video(source)
    
    session = config.DEFAULT_SESSION.copy()
    session.update({
        "status": SessionStatus.PROCESSING,
        "frames": [],
        "source": source,
        "cap": cap,
        "next_frame": 0,
        "chunk_sz": chunk_sz,
        "max_frames": max_frames
    })
    
    SESSIONS[session_id] = session
    
    return process_chunk(session_id)

@app.post("/continue")
def continue_processing(session_id: str = Form(...)):
    if session_id not in SESSIONS:
        return {"error": "Session not found"}
    
    if SESSIONS[session_id]["status"] == SessionStatus.DONE:
        return {"error": "Session is already done"}
    
    elif SESSIONS[session_id]["status"] == SessionStatus.FAILED:
        return {"error": "Session has failed"}
    
    elif SESSIONS[session_id]["status"] == SessionStatus.PROCESSING:
        return {"error": "Session is already processing"}
    
    elif SESSIONS[session_id]["status"] == SessionStatus.CANCELLED:
        return {"error": "Session was cancelled"}
    
    elif SESSIONS[session_id]["status"] == SessionStatus.PENDING:
        SESSIONS[session_id]["status"] = SessionStatus.PROCESSING
        return process_chunk(session_id)

    else:
        return {"error": "Session is not in a valid state for continuation"}

@app.post("/cancel/{session_id}")
def cancel_processing(session_id: str):
    if session_id not in SESSIONS:
        return {"error": "Session not found"}
    
    if SESSIONS[session_id]["status"] == SessionStatus.DONE:
        return {"error": "Session is already done"}

    elif SESSIONS[session_id]["status"] == SessionStatus.FAILED:
        return {"error": "Session has failed"}

    elif SESSIONS[session_id]["status"] == SessionStatus.PROCESSING:
        return {"error": "Session is already processing"}

    elif SESSIONS[session_id]["status"] == SessionStatus.CANCELLED:
        return {"error": "Session is already cancelled"}

    elif SESSIONS[session_id]["status"] == SessionStatus.PENDING:
        SESSIONS[session_id]["status"] = SessionStatus.CANCELLED
        response = session2response(session_id)
        
        clear_session(session_id, SESSIONS)
        return response

    else:
        return {"error": "Session is not in a valid state for cancellation"}

@app.get("/results/{session_id}")
def read_results(session_id: str):
    return session2response(session_id)
