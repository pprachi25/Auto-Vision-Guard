# AutoVision Guard — Pro v2.0

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```
Open: http://localhost:5000

## Features
- YOLOv8 real-time detection (Live camera, Image, Video)
- Advanced dashboard with 5 Chart.js charts
- Per-class detection breakdown with confidence scores
- Video frame-by-frame timeline chart
- Live camera stats (vehicles, pedestrians, risk) updated every 2s
- Session stats API (/api/live_stats)
- Risk level: LOW / MEDIUM / HIGH (computed from detections)
- Traffic status: SMOOTH / MODERATE / HEAVY
