#!/usr/bin/env python3
"""
Minimal YOLOSegTFLite demo:
- Loads a TFLite YOLO-seg model via detectors.YOLOSegTFLite
- Runs detection on a video / webcam / image
- Draws bounding boxes (and labels) + optional mask overlay
- Optional video writer

Usage examples:
  python detect_yolo_only.py --weights yoloseg.tflite --source input.mp4
  python detect_yolo_only.py --weights yoloseg.tflite --source 0 --show
  python detect_yolo_only.py --weights yoloseg.tflite --source image.jpg --show
  python detect_yolo_only.py --weights yoloseg.tflite --source input.mp4 --save out.mp4 --show
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --- only dependency from your repo ---
from yolo_tflite import YOLOSegTFLite  # must be available in PYTHONPATH


def parse_args():
    p = argparse.ArgumentParser("YOLOSegTFLite minimal detector")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO TFLite weights")
    p.add_argument("--source", type=str, required=True,
                   help="Video path, image path, or camera index (e.g. '0')")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--nc", type=int, default=None,
                   help="Number of classes (falls back to len(class-names) or 80)")
    p.add_argument("--show", action="store_true", help="Show live preview window")
    p.add_argument("--save", type=str, default=None, help="Optional output video path (MP4)")
    p.add_argument("--mask", action="store_true", help="Draw instance masks if provided by model")
    p.add_argument("--target-fps", type=float, default=None,
                   help="Force writer FPS (default: source FPS or 30 if unknown)")
    return p.parse_args()


def deterministic_color(idx: int) -> Tuple[int, int, int]:
    # BGR
    rng = np.random.RandomState(idx * 123457 % 2**31)
    return (int(rng.randint(50, 255)), int(rng.randint(50, 255)), int(rng.randint(50, 255)))


def load_class_names(txt_path: Optional[str], nc_fallback: int) -> List[str]:
    if txt_path is not None:
        lines = [ln.strip() for ln in Path(txt_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) > 0:
            return lines
    return [f"cls_{i}" for i in range(nc_fallback)]


def draw_boxes_and_masks(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clses: np.ndarray,
    masks_list: List[Optional[np.ndarray]],
    class_names: List[str],
    draw_masks: bool,
):
    H, W = frame.shape[:2]

    # Optional mask overlay (simple & robust)
    if draw_masks and masks_list is not None and len(masks_list) == len(boxes_xyxy):
        overlay = frame.copy()
        for i, m in enumerate(masks_list):
            if m is None:
                continue
            # m is expected to be 2D or 3D mask; resize to full frame if needed
            if m.ndim == 2:
                mask = m
            else:
                mask = m.squeeze()
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask > 0.5).astype(np.uint8)

            color = np.array(deterministic_color(int(clses[i])))[None, None, :]
            colored = np.zeros_like(frame, dtype=np.uint8)
            colored[:] = color
            overlay = np.where(mask_bin[..., None].astype(bool), (0.4 * colored + 0.6 * overlay).astype(np.uint8), overlay)

        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0.0, dst=frame)

    # Boxes + labels
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        cls_id = int(clses[i]) if i < len(clses) else 0
        conf = float(confs[i]) if i < len(confs) else 0.0
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id{cls_id}"
        color = deterministic_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)


def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    args = parse_args()

    # Determine class names and nc
    default_nc = 3
    class_names = ['person', 'animal', 'face']
    nc = len(class_names)

    # Build detector
    yolo = YOLOSegTFLite(
        args.weights,
        imgsz=(384,480),
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=nc
    )

    # Single image mode
    if is_image_file(args.source):
        img = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {args.source}")
        t0 = time.time()
        boxes, confs, clses, masks_list = yolo.detect(img)
        dt = (time.time() - t0) * 1000.0
        draw_boxes_and_masks(img, boxes, confs, clses, masks_list, class_names, draw_masks=args.mask)
        cv2.putText(img, f"{boxes.shape[0]} dets | {dt:.1f} ms", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{boxes.shape[0]} dets | {dt:.1f} ms", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if args.show:
            cv2.imshow("YOLOSegTFLite (image)", img)
            cv2.waitKey(0)
        if args.save:
            ok = cv2.imwrite(args.save, img)
            print(f"Saved: {args.save} ({'ok' if ok else 'failed'})")
        return

    # Video / webcam mode
    try:
        cam_index = int(args.source)
        source = cam_index
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # Get FPS and size
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 1e-3:
        in_fps = 30.0
    target_fps = args.target_fps if args.target_fps is not None else in_fps

    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Failed to read first frame")
    H, W = frame.shape[:2]

    # Optional writer
    writer = None
    if args.save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, float(target_fps), (W, H))
        if not writer.isOpened():
            print("[WARN] Could not open writer; disabling save.")
            writer = None

    # Main loop
    tic = time.time()
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            t0 = time.time()
            boxes, confs, clses, masks_list = yolo.detect(frame)
            infer_ms = (time.time() - t0) * 1000.0

            draw_boxes_and_masks(frame, boxes, confs, clses, masks_list, class_names, draw_masks=args.mask)

            # FPS overlay
            elapsed = time.time() - tic
            fps = frame_count / max(1e-9, elapsed)
            cv2.putText(frame, f"{boxes.shape[0]} dets | {infer_ms:.1f} ms | {fps:.1f} FPS",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{boxes.shape[0]} dets | {infer_ms:.1f} ms | {fps:.1f} FPS",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("YOLOSegTFLite (video)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer is not None:
                writer.write(frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Some environments benefit from disabling OpenCV thread oversubscription
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    main()
