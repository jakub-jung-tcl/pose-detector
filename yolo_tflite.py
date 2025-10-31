#!/usr/bin/env python3
"""
TFLite YOLO-seg (nms=False) inference module with post-processing (NumPy-only).

Exports a single class `YOLOSegTFLite` that you can call like:

    yoloseg = YOLOSegTFLite(weights, imgsz=(H, W), conf_thres=0.25, iou_thres=0.7, nc=3)
    xyxy, conf, cls, masks = yoloseg.detect(frame_bgr)

Where:
- `xyxy`  : np.float32 [N, 4] in ORIGINAL FRAME coordinates
- `conf`  : np.float32 [N]
- `cls`   : np.int32   [N]
- `masks` : List[bool HxW]
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple
import tensorflow as tf

# --------------------------
# Small helpers (NumPy)
# --------------------------
def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == 4, f"expected last dim=4, got {x.shape}"
    y = np.empty_like(x, dtype=np.float32)
    cx, cy = x[..., 0], x[..., 1]
    w2, h2 = x[..., 2] / 2.0, x[..., 3] / 2.0
    y[..., 0] = cx - w2  # x1
    y[..., 1] = cy - h2  # y1
    y[..., 2] = cx + w2  # x2
    y[..., 3] = cy + h2  # y2
    return y

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _greedy_nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int = 300) -> np.ndarray:
    """Return indices kept by NMS (NumPy)."""
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    order = scores.argsort()[::-1].astype(np.int64)
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
        union = areas[i] + areas[rest] - inter + 1e-12
        iou = inter / union

        inds = np.where(iou <= iou_thres)[0]
        order = rest[inds]

    return np.asarray(keep, dtype=np.int64)

def _nms_numpy(preds_nc: np.ndarray, nc: int, conf_thres: float, iou_thres: float,
               agnostic: bool = False, max_det: int = 300) -> np.ndarray:
    """
    NumPy NMS on YOLO (N, 4+nc+nm) predictions already in resized coordinates.

    Returns array shaped (K, 6+nm): [x1,y1,x2,y2, conf, cls_id, coeffs...]
    """
    if preds_nc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    preds_nc = preds_nc.astype(np.float32, copy=False)
    boxes_xyxy = _xywh2xyxy(preds_nc[:, :4])     # (N,4) in resized coords
    cls_scores = preds_nc[:, 4:4 + nc]           # (N,nc)
    coeffs     = preds_nc[:, 4 + nc:]            # (N,nm)
    nm = coeffs.shape[1] if coeffs.ndim == 2 else 0

    conf = cls_scores.max(axis=1)
    cls  = cls_scores.argmax(axis=1).astype(np.float32)  # keep float to mirror prior behavior

    # confidence threshold
    keep_mask = conf >= float(conf_thres)
    if not np.any(keep_mask):
        return np.zeros((0, 6 + nm), dtype=np.float32)

    boxes_xyxy = boxes_xyxy[keep_mask]
    conf = conf[keep_mask]
    cls_ids = cls[keep_mask]
    coeffs = coeffs[keep_mask] if nm > 0 else np.zeros((boxes_xyxy.shape[0], 0), dtype=np.float32)

    dets_list: List[np.ndarray] = []

    if agnostic:
        keep_idx = _greedy_nms(boxes_xyxy, conf, iou_thres, max_det=max_det)
        if keep_idx.size > 0:
            dets = np.concatenate(
                [boxes_xyxy[keep_idx],
                 conf[keep_idx, None],
                 cls_ids[keep_idx, None],
                 coeffs[keep_idx]], axis=1
            )
            dets_list.append(dets)
    else:
        # per-class NMS
        for c in np.unique(cls_ids.astype(np.int32)):
            m = (cls_ids == float(c))
            b_c = boxes_xyxy[m]
            s_c = conf[m]
            k_c = coeffs[m]
            if b_c.size == 0:
                continue
            keep_idx = _greedy_nms(b_c, s_c, iou_thres, max_det=max_det)
            if keep_idx.size == 0:
                continue
            dets = np.concatenate(
                [b_c[keep_idx],
                 s_c[keep_idx, None],
                 np.full((keep_idx.size, 1), float(c), dtype=np.float32),
                 k_c[keep_idx]], axis=1
            )
            dets_list.append(dets)

    if not dets_list:
        return np.zeros((0, 6 + nm), dtype=np.float32)

    dets_all = np.vstack(dets_list).astype(np.float32, copy=False)
    # sort final by score desc, clip to max_det total
    order = dets_all[:, 4].argsort()[::-1]
    dets_all = dets_all[order]
    if dets_all.shape[0] > max_det:
        dets_all = dets_all[:max_det]
    return dets_all

# --------------------------
# Raw TFLite wrapper
# --------------------------
class _TFLiteModel:
    def __init__(self, model_path: str, img_size: Tuple[int, int]):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.input_detail   = self.interp.get_input_details()[0]
        self.output_details = self.interp.get_output_details()
        self.img_size = img_size  # (H, W)
        self.in_dtype, self.in_quant = (
            self.input_detail["dtype"],
            self.input_detail["quantization"],
        )

    @staticmethod
    def _dequantize(arr, detail):
        out_dtype = detail["dtype"]
        if np.issubdtype(out_dtype, np.integer):
            scale_out, zp_out = detail["quantization"]
            arr = (arr.astype(np.float32) - zp_out) * scale_out
        else:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def _squeeze1(x):
        while x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x

    def predict(self, img_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (preds_nc, protos). preds_nc is (N, C). protos is (Hp, Wp, nm)."""
        inp = img_tensor.astype(np.float32)
        if np.issubdtype(self.in_dtype, np.integer):
            scale_in, zp_in = self.in_quant
            q = (inp / scale_in + zp_in).round()
            q = np.clip(q, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
            inp = q.astype(self.in_dtype)

        self.interp.set_tensor(self.input_detail["index"], inp)
        self.interp.invoke()

        outs = []
        for od in self.output_details:
            arr = self.interp.get_tensor(od["index"])
            outs.append(self._dequantize(self._squeeze1(arr), od))

        preds_nc, protos = None, None
        for o in outs:
            if o.ndim == 2:
                if o.shape[0] <= 256 and o.shape[1] > o.shape[0]:
                    o = o.T
                preds_nc = o.astype(np.float32)
            elif o.ndim == 3:
                if o.shape[2] <= 256:
                    protos = o.astype(np.float32)
                elif o.shape[0] <= 256:
                    protos = np.transpose(o, (1, 2, 0)).astype(np.float32)

        if preds_nc is None or protos is None:
            shapes = [getattr(x, 'shape', None) for x in outs]
            raise RuntimeError(f"Could not find preds/protos. Got {shapes}")

        # scale xywh (relative -> resized coords)
        H, W = self.img_size
        preds_nc[:, [0, 2]] *= W
        preds_nc[:, [1, 3]] *= H
        return preds_nc, protos

# --------------------------
# Public detector
# --------------------------
class YOLOSegTFLite:
    def __init__(self, weights: str, imgsz: Tuple[int, int], conf_thres: float = 0.25, iou_thres: float = 0.7, nc: int = 3):
        self.imgsz = tuple(imgsz)  # (H, W)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.nc = int(nc)
        self.model = _TFLiteModel(weights, img_size=self.imgsz)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Run detection + mask decode and return arrays in ORIGINAL frame coordinates.
        Returns (xyxy, conf, cls, masks_list).
        """
        H, W = frame_bgr.shape[:2]
        Hs, Ws = self.imgsz

        # --- preprocess (stretch)
        resized = cv2.resize(frame_bgr, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        ratio_h = Hs / max(H, 1)
        ratio_w = Ws / max(W, 1)
        img_in = resized[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB
        img_in = img_in[None]  # (1,H,W,3)

        # --- TFLite inference
        preds_nc, protos = self.model.predict(img_in)  # (N,C), (Hp,Wp,nm)
        nm = int(protos.shape[2])
        Hp, Wp, _ = protos.shape

        # --- NMS (NumPy)
        dets = _nms_numpy(preds_nc, nc=self.nc, conf_thres=self.conf_thres, iou_thres=self.iou_thres, agnostic=False)
        n_det = dets.shape[0]
        if n_det == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
                []
            )

        # --- build masks
        proto_mat = protos.reshape(-1, nm).astype(np.float32)  # (Hp*Wp, nm)
        masks_list: List[np.ndarray] = []

        # boxes on resized image -> original via ratio
        xyxy_resized = dets[:, :4].astype(np.float32)
        confs = dets[:, 4].astype(np.float32)
        clses = dets[:, 5].astype(np.int32)

        for i in range(n_det):
            x1r, y1r, x2r, y2r = xyxy_resized[i]
            coeffs = dets[i, 6:6+nm].astype(np.float32)

            # proto mask at proto scale
            m = proto_mat @ coeffs
            m = _sigmoid(m).reshape(Hp, Wp)

            # crop to bbox at proto scale
            sx = Wp / float(Ws)
            sy = Hp / float(Hs)
            x1m = int(max(0, min(Wp - 1, round(x1r * sx))))
            y1m = int(max(0, min(Hp - 1, round(y1r * sy))))
            x2m = int(max(0, min(Wp,     round(x2r * sx))))
            y2m = int(max(0, min(Hp,     round(y2r * sy))))
            crop_gate = np.zeros((Hp, Wp), dtype=np.float32)
            crop_gate[y1m:y2m, x1m:x2m] = 1.0
            m *= crop_gate

            # --- Single resize directly to ORIGINAL (keep INTER_LINEAR)
            m_orig_prob = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            m_orig_bin = (m_orig_prob >= 0.5).astype(bool)

            masks_list.append(m_orig_bin)

        # map boxes to original coords
        xyxy_orig = xyxy_resized.copy()
        xyxy_orig[:, [0, 2]] = xyxy_resized[:, [0, 2]] / max(ratio_w, 1e-12)
        xyxy_orig[:, [1, 3]] = xyxy_resized[:, [1, 3]] / max(ratio_h, 1e-12)

        return xyxy_orig.astype(np.float32), confs, clses, masks_list
