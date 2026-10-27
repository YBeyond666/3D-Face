from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from .data import read_rgb
from .io import write_jsonl


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, float)[:, :2])))
    if len(points) < 3:
        raise ValueError("At least three distinct points needed for face hull")
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_mask(points, height, width):
    image = Image.new("L", (width, height))
    ImageDraw.Draw(image).polygon(convex_hull(points), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.


def crop_matrix(landmarks, scale, size=224):
    xy = np.asarray(landmarks, float)[:, :2]
    if not np.isfinite(xy).all(): raise ValueError("Nonfinite landmarks")
    low, high = xy.min(0), xy.max(0)
    side = int(np.mean(high - low) * scale)
    if side < 2: raise ValueError("Degenerate face crop")
    origin = (high + low) / 2 - side / 2
    factor = (size - 1) / side
    return np.array([[factor, 0, -origin[0]*factor], [0, factor, -origin[1]*factor], [0, 0, 1.]])


def apply_crop(image, matrix, size=224, mask=False):
    inverse = np.linalg.inv(matrix)
    pil = Image.fromarray(np.uint8(np.clip(image, 0, 255)))
    result = pil.transform((size, size), Image.Transform.AFFINE, tuple(inverse[:2].ravel()),
                           resample=Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)
    return np.asarray(result)


def normalized_landmarks(points, matrix, size=224):
    p = np.asarray(points)[:, :2]
    cropped = np.c_[p, np.ones(len(p))] @ matrix.T
    return (cropped[:, :2] / size * 2 - 1).astype(np.float32)


class Detector:
    """FAN + MediaPipe Tasks, outputs original RGB pixel coordinates."""
    def __init__(self, mediapipe_asset, device="cpu"):
        import face_alignment
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if not Path(mediapipe_asset).is_file(): raise FileNotFoundError(mediapipe_asset)
        self.mp = mp
        enum = face_alignment.LandmarksType
        kind = getattr(enum, "TWO_D", None)
        if kind is None: kind = getattr(enum, "_2D")
        self.fan = face_alignment.FaceAlignment(kind, device=device, flip_input=False)
        self.detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(mediapipe_asset)), num_faces=2))

    def __call__(self, image):
        detections = self.fan.get_landmarks(image)
        media = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
        if detections is None or len(detections) != 1 or len(media.face_landmarks) != 1:
            raise ValueError("Expected exactly one face from both detectors; no automatic person selection")
        fan = np.asarray(detections[0], np.float32)[:, :2]
        h, w = image.shape[:2]
        mp = np.array([[p.x*w, p.y*h] for p in media.face_landmarks[0]], np.float32)
        if fan.shape != (68, 2) or len(mp) < 468 or not np.isfinite(fan).all() or not np.isfinite(mp).all():
            raise ValueError("Invalid detector result")
        return fan, mp

    def close(self): self.detector.close()


def preprocess(rows, output, detector):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    accepted, failures = [], []
    for r in rows:
        try:
            image = read_rgb(r)
            fan, mp = detector(image)
            key = hashlib.sha256(r["sample_id"].encode()).hexdigest()[:24]
            aux = output / f"{key}.npz"
            # Hull is a declared initial recipe, NOT an anatomical skin segmentation.
            np.savez_compressed(aux, fan=fan, mediapipe=mp, face_mask=hull_mask(mp, *image.shape[:2]),
                                mask_source=np.array("mediapipe_convex_hull"), image_hw=np.array(image.shape[:2]))
            accepted.append(dict(r, aux_path=str(aux), mask_source="mediapipe_convex_hull"))
        except (ValueError, OSError, RuntimeError) as exc:
            failures.append({"sample_id": r["sample_id"], "error": str(exc)})
    write_jsonl(output / "accepted.jsonl", accepted)
    write_jsonl(output / "failures.jsonl", failures)
    return {"accepted": len(accepted), "failed": len(failures), "manifest": str(output / "accepted.jsonl")}
