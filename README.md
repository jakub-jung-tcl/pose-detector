Head Pose Detector (Webcam)

Features
- Displays live camera feed
- Detects face and draws the bounding box
- Crops the face region and runs a high-accuracy landmark model (MediaPipe Face Mesh)
- Solves Perspective-n-Point (PnP) using a generic 3D facial template
- Computes and overlays yaw, pitch, and roll in degrees

Requirements
- Python 3.9+
- Packages: opencv-python, numpy, and either tensorflow or tflite-runtime

Install
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Models
- Place two TFLite models in the repo root:
  - `face_detection.tflite` — a face detector with TFLite_Detection_PostProcess outputs (e.g. MediaPipe BlazeFace short-range/front).
  - `face_landmark.tflite` — a FaceMesh-style landmark model (e.g. MediaPipe 468-landmark model).

Install
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install tensorflow  # or: pip install tflite-runtime
```

Run
```
python app.py
```

Notes
- If your webcam view is mirrored by your camera driver, yaw sign may appear inverted. You can adapt the euler-angle mapping if needed.
- The camera intrinsics are approximated using the frame width as focal length, which is sufficient for head-pose visualization.
- The app runs at ~4 FPS via a simple frame-rate limiter to reduce CPU usage.
