import uuid
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional

import app.config as config
from app.reid import ReID
from utils.detector_utils import load_model, connect_to_video
from utils.server_utils import process_video

app = FastAPI(title="Unique Object Tracking App")

REID = ReID()
MODEL = load_model(config.MODEL_PATH)
SESSIONS = {}

@app.get("/")
def read_root():
    return {"appname": "Unique Object Tracking App"}

@app.post("/process")
async def process(
    video_file: Optional[UploadFile] = File(None),
    stream_url: Optional[str] = Form(None),
    max_frames: int = Form(1000),
    ):

    session_id = str(uuid.uuid4())
    if video_file:
        video_path = f"tmp/{session_id}_{video_file.filename}"
        
        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        source = video_file.filename
        cap = cv2.VideoCapture(video_path)

        
        
    elif stream_url:
        source = stream_url
        cap = connect_to_video(stream_url)

    processed_frames = process_video(cap, MODEL, REID, tracker=config.TRACKER, skip_classes=config.SKIP_CLASSES, max_frames=max_frames)
    cap.release()
    
    result = {
        "session_id": session_id,
        "processed_frames": processed_frames,
        "source": source,
        }

    SESSIONS[session_id] = {
        "status": "done", 
        "frames": processed_frames, 
        "source": source,
        }
    return result

@app.get("/results/{session_id}")
def read_results(session_id: str):
    default_session = {
        "status": "unknown session",
        "frames": [],
        "source": None
    }
    
    return SESSIONS.get(session_id, default_session)
