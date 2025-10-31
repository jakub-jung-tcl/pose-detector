#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "TensorFlow (or tflite-runtime) is required for face detection TFLite.\n"
        "Install with `pip install tensorflow` or a platform-specific `tflite-runtime`.\n"
        + str(e)
    )


class FaceDetectorTFLite:
    """
    Supports two model families:
    1) TFLite_Detection_PostProcess style: boxes [1,N,4], scores [1,N], classes [1,N], (optional count)
    2) MediaPipe BlazeFace-style raw outputs: regressors [1,896,16], scores [1,896,1]

    Returns boxes in original image coordinates (x1, y1, x2, y2).
    """

    def __init__(self, model_path: str, conf_thres: float = 0.5):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.in_dtype = self.inp["dtype"]
        self.in_quant = self.inp.get("quantization", (0.0, 0))
        shp = self.inp.get("shape", None)
        if shp is None or len(shp) != 4:
            raise SystemExit(f"Unexpected input shape for detector: {shp}")
        self.Hs, self.Ws = int(shp[1]), int(shp[2])
        self.conf_thres = float(conf_thres)
        self._anchors_cache = None  # type: ignore

    @staticmethod
    def _quantize_if_needed(arr: np.ndarray, in_dtype, in_quant) -> np.ndarray:
        if np.issubdtype(in_dtype, np.integer):
            scale, zp = in_quant
            if scale == 0:
                scale = 1.0
            q = (arr / float(scale) + float(zp)).round()
            q = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max)
            return q.astype(in_dtype)
        return arr.astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5, max_det: int = 100) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes_xyxy.T
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0 and len(keep) < max_det:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
            inds = np.where(iou <= iou_thres)[0]
            order = rest[inds]
        return np.asarray(keep, dtype=np.int64)

    def _generate_blazeface_anchors(self, input_h: int, input_w: int) -> np.ndarray:
        """Generate 896 anchors for 128x128 input (BlazeFace short-range style).

        We use two feature maps: 16x16 (stride 8) with 2 anchors per cell,
        and 8x8 (stride 16) with 6 anchors per cell. Anchor centers are on the grid
        cell centers. Anchor sizes are square, with scales tuned for face sizes.
        """
        # Feature maps and anchors-per-cell
        fmap_specs = [
            (16, 16, 8, [0.10, 0.20]),                 # small feature map (stride 8)
            (8, 8, 16, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]),  # stride 16
        ]
        anchors = []
        for gh, gw, stride, scales in fmap_specs:
            for y in range(gh):
                for x in range(gw):
                    cy = (y + 0.5) / gh
                    cx = (x + 0.5) / gw
                    for s in scales:
                        anchors.append([cy, cx, s, s])  # y_center, x_center, h, w (normalized)
        a = np.asarray(anchors, dtype=np.float32)
        assert a.shape[0] == 896, f"Expected 896 anchors, got {a.shape[0]}"
        return a

    def _decode_blazeface(self, raw_boxes: np.ndarray, raw_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode BlazeFace raw outputs to normalized xyxy and scores.

        raw_boxes:  [896, 16]
        raw_scores: [896, 1] or [896]
        """
        if raw_scores.ndim == 2 and raw_scores.shape[1] == 1:
            raw_scores = raw_scores[:, 0]
        scores = self._sigmoid(raw_scores.astype(np.float32))

        # Build or reuse anchors
        if self._anchors_cache is None:
            self._anchors_cache = self._generate_blazeface_anchors(self.Hs, self.Ws)
        anc = self._anchors_cache  # (896, 4): [cy, cx, h, w] normalized

        # Scales from MediaPipe for short-range face detection
        y_scale = x_scale = h_scale = w_scale = 128.0

        # Extract first 4 values per anchor: ty, tx, th, tw
        t = raw_boxes[:, :4].astype(np.float32)
        ty, tx, th, tw = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

        # Decode
        y_center = ty / y_scale * anc[:, 2] + anc[:, 0]
        x_center = tx / x_scale * anc[:, 3] + anc[:, 1]
        h = np.exp(th / h_scale) * anc[:, 2]
        w = np.exp(tw / w_scale) * anc[:, 3]

        y1 = y_center - h * 0.5
        x1 = x_center - w * 0.5
        y2 = y_center + h * 0.5
        x2 = x_center + w * 0.5

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        # Clip to [0,1]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
        return boxes.astype(np.float32), scores.astype(np.float32)

    def _map_to_original(self, boxes_norm_xyxy: np.ndarray, W: int, H: int,
                          Ws: int, Hs: int, pad_left: int, pad_top: int, scale: float) -> np.ndarray:
        if boxes_norm_xyxy.size == 0:
            return boxes_norm_xyxy.astype(np.float32)
        b = boxes_norm_xyxy.astype(np.float32).copy()
        # to letterboxed pixel coords
        b[:, [0, 2]] = b[:, [0, 2]] * float(Ws) - float(pad_left)
        b[:, [1, 3]] = b[:, [1, 3]] * float(Hs) - float(pad_top)
        # to original image coords
        s = max(scale, 1e-12)
        b[:, [0, 2]] /= s
        b[:, [1, 3]] /= s
        # clip
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(W))
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(H))
        return b

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = frame_bgr.shape[:2]
        # letterbox to model input size
        scale = min(self.Ws / max(W, 1e-6), self.Hs / max(H, 1e-6))
        new_W = int(round(W * scale))
        new_H = int(round(H * scale))
        pad_left = (self.Ws - new_W) // 2
        pad_top = (self.Hs - new_H) // 2
        canvas = np.zeros((self.Hs, self.Ws, 3), dtype=np.uint8)
        if new_W > 0 and new_H > 0:
            resized = cv2.resize(frame_bgr, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            canvas[pad_top:pad_top + new_H, pad_left:pad_left + new_W] = resized
        img_in = canvas[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB
        img_in = img_in[None]
        img_in = self._quantize_if_needed(img_in, self.in_dtype, self.in_quant)

        self.interp.set_tensor(self.inp["index"], img_in)
        self.interp.invoke()

        # Inspect outputs
        def _dequantize(arr, od):
            if np.issubdtype(arr.dtype, np.integer):
                scale, zp = od.get("quantization", (0.0, 0))
                if scale and scale != 0.0:
                    return (arr.astype(np.float32) - float(zp)) * float(scale)
                return arr.astype(np.float32)
            return arr.astype(np.float32)

        out_tensors = []
        for od in self.outs:
            arr = self.interp.get_tensor(od["index"])
            out_tensors.append(_dequantize(arr, od))
        shapes = [getattr(t, 'shape', None) for t in out_tensors]

        boxes_xyxy = None
        scores = None
        clses = None

        # Case 1: PostProcess-style
        if any(len(s) == 3 and s[-1] == 4 for s in shapes if s is not None) and any(len(s) == 2 for s in shapes if s is not None):
            boxes_n = None
            for t in out_tensors:
                if t.ndim == 3 and t.shape[-1] == 4:
                    boxes_n = t.astype(np.float32).reshape(-1, 4)
                elif t.ndim == 2 and t.size == t.shape[0] * t.shape[1]:
                    # Heuristic: float array is scores, integer is classes
                    if t.dtype.kind in ('f', 'c'):
                        scores = t.astype(np.float32).reshape(-1)
                    else:
                        clses = t.astype(np.float32).reshape(-1)
            if boxes_n is None or scores is None:
                raise RuntimeError(f"Detector outputs not recognized (PostProcess path). Have {shapes}")

            keep = scores >= self.conf_thres
            boxes_n = boxes_n[keep]
            scores = scores[keep]
            clses = (np.zeros_like(scores) if clses is None else clses[keep]).astype(np.int32)

            if boxes_n.size == 0:
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                )

            # Convert [ymin, xmin, ymax, xmax] -> normalized xyxy
            boxes_norm_xyxy = np.stack([boxes_n[:, 1], boxes_n[:, 0], boxes_n[:, 3], boxes_n[:, 2]], axis=1)
            # Map to original image coords via letterbox inverse
            boxes_xyxy = self._map_to_original(boxes_norm_xyxy, W, H, self.Ws, self.Hs, pad_left, pad_top, scale)
            # NMS in image coords
            keep_idx = self._nms(boxes_xyxy, scores, iou_thres=0.5, max_det=100)
            boxes_xyxy = boxes_xyxy[keep_idx]
            scores = scores[keep_idx]
            clses = clses[keep_idx]

        # Case 2: BlazeFace raw tensors (e.g., [1,896,16], [1,896,1])
        elif any(s is not None and len(s) == 3 and s[1] == 896 and s[2] in (16, 12) for s in shapes):
            # raw_boxes: last dim 16, raw_scores: last dim 1
            raw_boxes = None
            raw_scores = None
            for t in out_tensors:
                if t.ndim == 3 and t.shape[1] == 896 and t.shape[2] >= 4:
                    raw_boxes = t.reshape(-1, t.shape[2]).astype(np.float32)
                if t.ndim == 3 and t.shape[1] == 896 and t.shape[2] == 1:
                    raw_scores = t.reshape(-1).astype(np.float32)
                if t.ndim == 2 and t.shape[0] == 1 and t.shape[1] == 896:
                    raw_scores = t.reshape(-1).astype(np.float32)

            if raw_boxes is None or raw_scores is None:
                raise RuntimeError(f"Could not parse BlazeFace outputs. Have {shapes}")

            boxes_n, scores = self._decode_blazeface(raw_boxes, raw_scores)
            # Threshold and NMS in normalized coords
            keep = scores >= self.conf_thres
            boxes_n = boxes_n[keep]
            scores = scores[keep]
            clses = np.zeros_like(scores, dtype=np.int32)

            if boxes_n.size == 0:
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                )

            # Map normalized boxes to original image coords via letterbox inverse
            boxes_xyxy = self._map_to_original(boxes_n, W, H, self.Ws, self.Hs, pad_left, pad_top, scale)
            # NMS in image coords
            keep_idx = self._nms(boxes_xyxy, scores, iou_thres=0.5, max_det=100)
            boxes_xyxy = boxes_xyxy[keep_idx]
            scores = scores[keep_idx]
            clses = clses[keep_idx]

        else:
            raise RuntimeError(f"Detector outputs not recognized. Have {shapes}")

        return boxes_xyxy.astype(np.float32), scores.astype(np.float32), clses.astype(np.int32)
