import os
import cv2
import requests
import app.config as config
from argparse import ArgumentParser

from utils.detector_utils import draw_boxes, connect_to_video

def send_video(file_path, max_frames=1000, chunk_sz=200):
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    with open(file_path, "rb") as f:
        video_file = f.read()

    response = requests.post(f"{config.SERVER_URL}/process", files={"video_file": video_file}, data={
        "chunk_sz": chunk_sz,
        "max_frames": max_frames
    })

    return response.json()

def continue_session(session_id):
    response = requests.post(f"{config.SERVER_URL}/continue", data={
        "session_id": session_id
    })    
    return response.json()

def cancel_session(session_id):
    response = requests.post(f"{config.SERVER_URL}/cancel/{session_id}")

    print("Response: ", response)
    # return response.json()

def build_output_video(input_path, results, output_path):
    if not os.path.exists(input_path):
        print("Input video not found.")
        return

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    cap = connect_to_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened() and frame_idx < len(results):
        success, frame = cap.read()
        if not success:
            break

        if frame_idx != results[frame_idx]["frame_index"]:
            continue
        
        detections = results[frame_idx]["detections"]
        if detections:
            draw_boxes(frame, detections)

        out.write(frame)
        frame_idx += 1
    
    print("Frame_idx: ", frame_idx)

    cap.release()
    out.release()
    print(f"Output video saved at {output_path}")

def main(video_path, chunk_sz=200, max_frames=1000):
    results = []

    print("Sending video for processing...")
    response = send_video(video_path, chunk_sz=chunk_sz, max_frames=max_frames)
    session_id = response["session_id"]
    results = response["processed_frames"]

    while response["status"] == config.SessionStatus.PENDING.value:
        user_input = input("Continue processing? (y/n): ").strip().lower()
        
        if user_input != "y":
            cancel_session(session_id)
            print("Processing cancelled.")
            break
        
        response = continue_session(session_id)
        print("Continue session response: ", response.keys())
        print("Processed frames: ", len(response.get("processed_frames")))
        results = response["processed_frames"]

    if response["status"] in [config.SessionStatus.DONE.value, config.SessionStatus.CANCELLED.value]:
        print("Processing completed successfully!")

    output_path = os.path.join(config.RESULT_FOLDER, f"{session_id[:8]}.mp4")
    
    build_output_video(video_path, results, output_path=output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--chunk_size", type=int, default=400, help="Size of each processing chunk")
    parser.add_argument("--max_frames", type=int, default=1000, help="Maximum number of frames to process")
    
    args = parser.parse_args()

    main(args.video_path, args.chunk_size, args.max_frames)
