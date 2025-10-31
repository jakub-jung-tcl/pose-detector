"""
Sources for TFLite models used:

- face_detection.tflite (MediaPipe BlazeFace short-range)
  Primary: https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
  Mirror : https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_detection/face_detection_short_range.tflite

- face_landmark.tflite (MediaPipe Face Mesh, 468 landmarks)
  Primary: https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
  Mirror : https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_landmark/face_landmark.tflite
"""

import math
from typing import Tuple, Optional

import cv2
import numpy as np
import time

# Raw TFLite backends
from face_detection_tflite import FaceDetectorTFLite
from facemesh_tflite import FaceMeshTFLite


# FaceMesh landmark indices for key points
# Reference (MediaPipe Face Mesh 468):
#  - Nose tip: 1
#  - Chin: 152
#  - Left eye outer corner: 33
#  - Right eye outer corner: 263
#  - Left mouth corner: 61
#  - Right mouth corner: 291
LMK_NOSE_TIP = 1
LMK_CHIN = 152
LMK_LEFT_EYE_OUT = 33
LMK_RIGHT_EYE_OUT = 263
LMK_LEFT_MOUTH = 61
LMK_RIGHT_MOUTH = 291

# BBox thresholding for landmark reliability
MIN_BBOX_SIDE_FRAC = 0.30  # minimum fraction of min(image side)
MIN_BBOX_SIDE_PX = 120     # absolute minimum in pixels

# Target FPS limiter
TARGET_FPS = 30.0


def get_camera_matrix(frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame_shape
    focal_length = w  # reasonable approximation
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def solve_head_pose(image_points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 3D model points for a generic head (in mm)
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Nose tip
            [0.0, -330.0, -65.0],  # Chin
            [-225.0, 170.0, -135.0],  # Left eye outer corner
            [225.0, 170.0, -135.0],  # Right eye outer corner
            [-150.0, -150.0, -125.0],  # Left Mouth corner
            [150.0, -150.0, -125.0],  # Right mouth corner
        ],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, rvec, tvec


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    # Following OpenCV convention (right-handed, x-right, y-down, z-forward)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2, 1], R[2, 2]))  # pitch
        y = math.degrees(math.atan2(-R[2, 0], sy))  # yaw
        z = math.degrees(math.atan2(R[1, 0], R[0, 0]))  # roll
    else:
        x = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = 0.0
    return y, x, z  # return as yaw, pitch, roll


def landmarks_to_image_points(landmarks, x0: int, y0: int, scale_w: int, scale_h: int) -> Optional[np.ndarray]:
    if not landmarks:
        return None
    lms = landmarks.landmark
    idxs = [LMK_NOSE_TIP, LMK_CHIN, LMK_LEFT_EYE_OUT, LMK_RIGHT_EYE_OUT, LMK_LEFT_MOUTH, LMK_RIGHT_MOUTH]
    pts = []
    for idx in idxs:
        lm = lms[idx]
        x = lm.x * scale_w + x0
        y = lm.y * scale_h + y0
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)


def enlarge_bbox(x: int, y: int, w: int, h: int, scale: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * scale
    new_h = h * scale
    x1 = int(max(0, round(cx - new_w / 2.0)))
    y1 = int(max(0, round(cy - new_h / 2.0)))
    x2 = int(min(img_w, round(cx + new_w / 2.0)))
    y2 = int(min(img_h, round(cy + new_h / 2.0)))
    return x1, y1, x2 - x1, y2 - y1


def draw_pose_text(frame: np.ndarray, x: int, y: int, yaw: float, pitch: float, roll: float):
    text = f"yaw:{yaw:+.1f}  pitch:{pitch:+.1f}  roll:{roll:+.1f}"
    cv2.rectangle(frame, (x, max(0, y - 24)), (x + 260, y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 6, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def default_landmark_points_center(img_w: int, img_h: int) -> np.ndarray:
    # Arrange 6 points around the image center in a face-like pattern
    cx, cy = img_w // 2, img_h // 2
    dx_eye, dy_eye = 60, 50
    dx_mouth, dy_mouth = 45, 45
    chin_off = 70
    pts = [
        [cx, cy],  # nose
        [cx, cy + chin_off],  # chin
        [cx - dx_eye, cy - dy_eye],  # left eye outer
        [cx + dx_eye, cy - dy_eye],  # right eye outer
        [cx - dx_mouth, cy + dy_mouth],  # left mouth
        [cx + dx_mouth, cy + dy_mouth],  # right mouth
    ]
    return np.asarray(pts, dtype=np.float64)


def draw_landmark_points(frame: np.ndarray, points: np.ndarray, color=(255, 0, 0)):
    for px, py in points:
        cv2.circle(frame, (int(px), int(py)), 2, color, -1)


def draw_default_face_ellipse(frame: np.ndarray, img_w: int, img_h: int, color=(0, 255, 255), thickness: int = 2):
    # Draw an ellipse centered on the screen resembling a face outline
    cx, cy = img_w // 2, img_h // 2
    s = min(img_w, img_h)
    axes = (int(s * 0.18), int(s * 0.24))  # horizontal and vertical radii
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, color, thickness)


def classify_pose(yaw: float, pitch: float, roll: float) -> Optional[str]:
    if abs(pitch) > 170 and abs(yaw) < 30 and (abs(roll) < 15 or abs(roll) > 165):
        return "Pose: front"
    if abs(pitch) > 150 and yaw > 45 and (abs(roll) < 20 or abs(roll) > 160):
        return "Pose: left"
    if abs(pitch) > 150 and yaw < -45 and (abs(roll) < 20 or abs(roll) > 160):
        return "Pose: right"
    return None


def draw_pose_label(frame: np.ndarray, x: int, y: int, label: str):
    # Draw a label above the angles text box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), bl = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    # Place this box above the existing angles box (which spans [y-24, y])
    bottom = max(0, y - 26)
    top = max(0, bottom - (th + 2 * pad))
    right = x + tw + 2 * pad
    cv2.rectangle(frame, (x, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + pad, bottom - pad - bl // 2), font, scale, (0, 255, 255), 2, cv2.LINE_AA)


def draw_axes(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length: float = 150.0, thickness: int = 2):
    # Axis in the head (object) coordinate system: X (red), Y (green), Z (blue)
    axis_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    o = tuple(imgpts[0])
    cv2.line(frame, o, tuple(imgpts[1]), (0, 0, 255), thickness)   # X - red
    cv2.line(frame, o, tuple(imgpts[2]), (0, 255, 0), thickness)   # Y - green
    cv2.line(frame, o, tuple(imgpts[3]), (255, 0, 0), thickness)   # Z - blue


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0")

    # Initialize raw TFLite detectors
    face_det = FaceDetectorTFLite("face_detection.tflite", conf_thres=0.5)
    face_mesh = FaceMeshTFLite("face_landmark.tflite", input_size=192)

    bbox_scale = 1  # enlarge bbox for cropping
    frame_interval = 1.0 / max(TARGET_FPS, 1e-6)

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        camera_matrix, dist_coeffs = get_camera_matrix((h, w))

        # Face detection via TFLite detector (boxes in image coords)
        boxes, confs, clses = face_det.detect(frame)

        if boxes.shape[0] > 0:
            # Pick highest score detection
            i_best = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[i_best]
            x = int(x1)
            y = int(y1)
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            x, y, bw, bh = enlarge_bbox(x, y, bw, bh, bbox_scale, w, h)
            x2 = x + bw
            y2 = y + bh

            # Draw face bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # Check bbox size; if too small, show stationary default landmarks in screen center
            min_side_needed = max(int(min(w, h) * MIN_BBOX_SIDE_FRAC), MIN_BBOX_SIDE_PX)
            if min(bw, bh) < min_side_needed:
                default_pts = default_landmark_points_center(w, h)
                draw_landmark_points(frame, default_pts)
                draw_default_face_ellipse(frame, w, h)
                cv2.imshow("Head Pose (press q to quit)", frame)
                elapsed = time.time() - loop_start
                delay_ms = int(max(1, round(max(0.0, frame_interval - elapsed) * 1000)))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue

            # Build aligned ROI (affine) centered at bbox center
            roi_size = 192
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            box_side = float(max(bw, bh))
            angle_deg = 0.0  # if you estimate roll, set it here
            scale = roi_size / max(1.0, box_side)
            M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
            # Translate so that (cx, cy) maps to ROI center
            M[0, 2] += (roi_size * 0.5 - cx)
            M[1, 2] += (roi_size * 0.5 - cy)

            roi_bgr = cv2.warpAffine(
                frame, M, (roi_size, roi_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            # Face landmarks via TFLite on aligned ROI
            lm_roi = face_mesh.detect(roi_bgr)  # (N,2) in ROI pixels

            # Map selected landmarks back to full image
            M3 = np.vstack([M, [0, 0, 1]])
            Minv = np.linalg.inv(M3)[:2, :]
            idxs = [LMK_NOSE_TIP, LMK_CHIN, LMK_LEFT_EYE_OUT, LMK_RIGHT_EYE_OUT, LMK_LEFT_MOUTH, LMK_RIGHT_MOUTH]
            if lm_roi.shape[0] <= max(idxs):
                image_points = None
            else:
                sel = lm_roi[idxs, :].astype(np.float64)
                ones = np.ones((sel.shape[0], 1), dtype=np.float64)
                pts_img = (Minv @ np.hstack([sel, ones]).T).T  # (6,2)
                image_points = pts_img

            if image_points is not None and np.isfinite(image_points).all():
                try:
                    R, rvec, tvec = solve_head_pose(image_points, camera_matrix, dist_coeffs)
                    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
                    draw_pose_text(frame, x, y, yaw, pitch, roll)
                    pose_label = classify_pose(yaw, pitch, roll)
                    if pose_label:
                        draw_pose_label(frame, x, y, pose_label)
                    # Draw 3D axes for visual verification of pose
                    draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, length=150, thickness=2)
                except Exception:
                    pass

            # Optionally draw the selected landmark points
            if image_points is not None:
                draw_landmark_points(frame, image_points)
        else:
            # No detection: also show stationary default landmarks in center
            default_pts = default_landmark_points_center(w, h)
            draw_landmark_points(frame, default_pts)

        cv2.imshow("Head Pose (press q to quit)", frame)
        elapsed = time.time() - loop_start
        delay_ms = int(max(1, round(max(0.0, frame_interval - elapsed) * 1000)))
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
