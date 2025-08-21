# Unique Object Tracking & Re‑Identification

Real‑time multi‑object tracking with YOLO / RT-DETR plus a lightweight ReID (person / object appearance embedding) layer backed by a persistent Chroma vector database. The system assigns stable IDs across frames (and future sessions) by comparing cropped detections to previously stored embeddings. Has two modes: local tracker UI or FastAPI server + client.

## Key Features
- YOLO / RT-DETR detection + built‑in tracker (BoT-SORT / ByteTrack)
- Appearance ReID using Torchreid backbones (default `resnet50` / configurable)
- Persistent vector DB (Chroma, DuckDB+Parquet) for cross‑run ID continuity
- Configurable similarity metric & confidence threshold (`cosine` is default)
- Support for local webcam or YouTube livestream input
- Easy pluggable model weights (place under `weights/`)
- Per-frame conflict resolution (pool): if multiple detections want the same id, only the most similar keeps it, others get new ids.
- API for chunked video processing and a client that rebuilds an annotated video (currently works only for local videos, meanwhile local tracker can show previewed results for online streams).

## Project Structure
```
app/
  config.py             # Loads .env into typed config values
  reid.py               # ReID embedding + vector DB logic
  tracker.py            # Main entry for tracking + ReID persistence
client/
  client.py             # Sends video to server, polls continuation, rebuilds annotated MP4.
server/
  server.py             # FastAPI endpoints: /process, /continue, /cancel/{id}, /results/{id}. Streams frames in chunks, persists session JSON to tmp/sessions.
utils/
  detector_utils.py     # Model loading, stream connect, box processing helpers
  server_utils.py       # Process frames and returns server compatible output
weights/                # Place YOLO / RT-DETR .pt files here
.vdb/                   # (Created at runtime) Chroma DB storage
.env                    # Environment configuration
```

## Requirements
Python 3.11+ recommended.

Core dependencies (see `requirements.txt`):
- `ultralytics` (YOLO & RT-DETR)
- `torch`, `torchvision`, `torchreid`
- `chromadb`
- `yt-dlp` (YouTube stream handling)
- `python-dotenv`
- `fastapi[standart]`
- `uvicorn[standart]`

## Installation
```powershell
# (Optional) Create & activate virtual environment
python -m venv .venv
./.venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```
If `torch` + CUDA is desired, install a wheel matching your GPU / CUDA version from https://pytorch.org beforehand (then `pip install -r requirements.txt` for the rest).

## Configuration (.env)

Key settings:
- STREAM_URL=tmp/vid1.mp4 (file path, camera index like 0, or YouTube URL)
- MODEL_PATH=weights/yolo11m.pt
- TRACKER=botsort.yaml
- COCO_NUM_CLASSES=80, CLASSES_THRESHOLD=8
  - SKIP_CLASSES is computed as range(CLASSES_THRESHOLD, COCO_NUM_CLASSES)
- CHROMADB_PATH=./.vdb/objects_ids
- REID_MODEL=osnet_x1_0
- REID_INPUT_SHAPE=256, 128
- REID_DISTANCE=cosine
- REID_SIM_THRESHOLD=0.7

To reset the vector DB, delete the .vdb/ folder or change CHROMADB_PATH.

### Similarity & Threshold
The ReID layer produces L2‑normalized embeddings. Depending on `REID_DISTANCE`:
- `cosine`: Chroma returns distance = 1 - cos_sim. We convert to confidence = cos_sim.
- `l2`: Max L2 distance between two normalized vectors ≈ √2. Confidence = 1 - d/√2.
- `ip`: Inner product (≈ cosine for normalized vectors). Confidence roughly mapped into [0,1].

`REID_SIM_THRESHOLD` is then compared against the derived confidence. Example: 0.7 means "only reuse an existing ID if we are ≥70% sure".

Tuning tips:
- Too many new IDs? Lower threshold (e.g. 0.6).
- Wrong merges (different objects share ID)? Raise threshold (e.g. 0.8–0.9).
- Consider filtering matches by the object label (logic hook available in `process_boxes`).

## Running
### FastAPI server:
```powershell
uvicorn server.server:app --host 127.0.0.1 --port 8000
```
### Client (build annotated video):
```powershell
python -m client.client path\to\input.mp4 --chunk_size 400 --max_frames 1000
```
Outputs to tmp/results/<session>.mp4.

### Local tracker (with utilities & class threshold)
```powershell
python -m app.tracker
```
Press `q` to quit the window.

### Using a YouTube Stream
Set `STREAM_URL` to the full YouTube video URL in `.env`. The helper resolves the direct stream via `yt-dlp`.

## Adding / Switching Detection Models
Place the `.pt` weight file under `weights/` and update `MODEL_PATH` in `.env`.
Supported automatically:
- YOLO (e.g. `yolov8n.pt`, `yolo11m.pt`)
- RT-DETR (e.g. `rtdetr-l.pt`)

## Persistent ReID Store
Chroma DB files are written under `CHROMADB_PATH` (ignored by git). Deleting that folder resets identity memory.

To inspect IDs after a run, the app logs them at shutdown. You can also add:
```python
print(re_id.count(), re_id.list_ids()[:5])
```
anywhere after initialization for runtime inspection.

## Customizing ReID
In `app/reid.py` you can:
- Change backbone (`REID_MODEL`): see Torchreid's model zoo (e.g., `osnet_x1_0`, `resnet50`, `mobilenetv2_x1_0`).
- Adjust input size: `REID_INPUT_SHAPE` (taller improves aspect fidelity but costs time).
- Switch metric: set `REID_DISTANCE` & delete existing DB folder to rebuild index under new metric.

## Per-frame Conflict Resolution (ids pool)
In utils/detector_utils.process_boxes:
- All detections are embedded in a batch.
- For each detection, we search the DB to get a candidate id (if any).
- If multiple detections in the same frame want the same id, we keep the one closest to the DB embedding (highest similarity). Others are forced to become new ids and get appended to DB.
- Drawing happens after all detections for the frame are processed to avoid mid-frame inconsistencies.

This prevents two different objects in the same frame from sharing a id.

## API
- POST /process
  - Form fields: stream_url (optional), video_file (optional), max_frames, chunk_sz
  - Starts a session and returns initial processed frames
- POST /continue
  - Form: session_id
  - Continues processing next chunk
- POST /cancel/{session_id}
  - Cancels and finalizes session; temp files cleaned
- GET /results/{session_id}
  - Returns saved session JSON

Responses are also saved to tmp/sessions/{session_id}.json.

## General Tips
- rt-detr finds more objects, while yolo11 shows visually and semantically better overal experience

## Performance Tips
- Use a GPU (CUDA) build of PyTorch for real‑time performance.
- Smaller detection models (e.g. `yolov8n`) trade accuracy for speed.
- Raise `REID_INPUT_SHAPE` for possibly better discrimination (slower) or lower it to speed up.
- Skip tiny detections (add a min box size check before generating embeddings) to reduce noise.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Import warnings in editor | VS Code not using venv | Select interpreter: Command Palette → Python: Select Interpreter |
| ReID always creates new IDs | Threshold too high / poor crops | Lower `REID_SIM_THRESHOLD`, ensure good lighting, skip tiny boxes |
| Wrong merges | Threshold too low | Increase threshold (0.8+), consider label filtering |
| Large distance spikes | Missing normalization | Ensure current `reid.py` has L2 normalize (already included) |
| No persistence | Wrong `CHROMADB_PATH` or deleted folder | Verify path & write permissions |

## Notes
- YouTube streaming uses yt-dlp to resolve a direct URL; make sure yt-dlp works in your environment.
- Skip tiny crops if you get noisy IDs (e.g., width < 16 or height < 32).
- To audit the DB, call print_vdb_info(re_id) or inspect tmp/sessions JSON outputs.

## Extending
Ideas:
- Store per‑ID exemplar images alongside embeddings.
- Periodically re-cluster embeddings to consolidate duplicates.
- Add temporal smoothing (Kalman) on top of tracker IDs.
- Export metrics (ID switches, MOTA) for evaluation.

## License
MIT.

## Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [Chroma](https://github.com/chroma-core/chroma)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

---
Feel free to open issues / tweak thresholds as you iterate on accuracy vs. stability.
