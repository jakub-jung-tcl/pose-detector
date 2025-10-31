#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple

import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "TensorFlow (or tflite-runtime) is required for FaceMesh TFLite.\n"
        "Install with `pip install tensorflow` or a platform-specific `tflite-runtime`.\n"
        + str(e)
    )


class FaceMeshTFLite:
    """
    Minimal TFLite FaceMesh wrapper.

    Expects a single image input (H, W, 3) and returns landmarks.
    Output is reshaped to (N, 3) if divisible by 3, and converted to ROI pixel coords.
    """

    def __init__(self, model_path: str, input_size: int = 192):
        self.input_size = int(input_size)
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.in_dtype = self.inp["dtype"]
        self.in_quant = self.inp["quantization"]  # (scale, zero_point) or (0.0, 0)

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

    def _select_landmark_output(self) -> int:
        # Pick the output that looks like flat landmark vector
        best_idx = 0
        best_score = -1
        for i, od in enumerate(self.outs):
            shape = od.get("shape", None)
            if shape is None:
                continue
            size = int(np.prod(shape))
            # Prefer vectors divisible by 3 (x,y,z per landmark)
            score = 0
            if size % 3 == 0:
                score += 2
            if size in (1404, 1434, 1602, 478*3):  # known variants
                score += 3
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def detect(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return landmarks as (N, 2) in ROI pixel coordinates."""
        Hs = Ws = self.input_size
        # Preprocess
        resized = cv2.resize(roi_bgr, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        img_in = resized[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0..1
        img_in = img_in[None]  # (1,H,W,3)
        img_in = self._quantize_if_needed(img_in, self.in_dtype, self.in_quant)

        # Inference
        self.interp.set_tensor(self.inp["index"], img_in)
        self.interp.invoke()

        out_idx = self._select_landmark_output()
        lm_arr = self.interp.get_tensor(self.outs[out_idx]["index"])  # unknown shape
        lm_vec = lm_arr.reshape(-1).astype(np.float32)
        if lm_vec.size % 3 != 0:
            raise RuntimeError(f"Unexpected FaceMesh output size: {lm_vec.size}")
        lm = lm_vec.reshape(-1, 3)

        # Convert to ROI pixel coords robustly
        xy = lm[:, :2].copy()
        mn, mx = float(xy.min(initial=0.0)), float(xy.max(initial=0.0))

        if -1.2 <= mn and mx <= 1.2:
            # [-1, 1] -> pixels
            xy = (xy * 0.5 + 0.5) * float(Ws)
        elif 0.0 <= mn and mx <= 1.01:
            # [0, 1] -> pixels
            xy = xy * float(Ws)
        else:
            # Assume already in pixels (common for some exports)
            pass

        return xy.astype(np.float32)

