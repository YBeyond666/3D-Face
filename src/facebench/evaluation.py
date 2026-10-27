from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import identity


def load_points(row):
    if row.get("geometry_unit") != "mm" or not row.get("geometry_frame"):
        raise ValueError("Point scoring requires explicit mm and geometry_frame metadata")
    with np.load(row["points_path"], allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], float)
        valid = np.asarray(data["valid"], bool)
        ids = [str(v) for v in data["point_ids"]]
    if xyz.shape != (len(ids), 3) or valid.shape != (len(ids),) or len(set(ids)) != len(ids):
        raise ValueError("Invalid point array or duplicate point IDs")
    if not np.isfinite(xyz[valid]).all():
        raise ValueError("Valid points contain NaN/Inf")
    return xyz, valid, ids


def evaluate_points(references, predictions, protocol):
    if protocol.get("coordinate_reviewed") is not True or not protocol.get("reference_description"):
        raise ValueError("Review coordinates/correspondence and describe reference before 3D scoring")
    expected_frame = protocol["geometry_frame"]
    threshold = float(protocol.get("success_threshold_mm", 5))
    if threshold <= 0:
        raise ValueError("Success threshold must be positive")
    pred = {}
    for row in predictions:
        if row["sample_id"] in pred:
            raise ValueError("Duplicate prediction sample_id")
        pred[row["sample_id"]] = row
    subjects, records, seen = defaultdict(list), [], set()
    for r in references:
        sid = r["sample_id"]
        if sid in seen:
            raise ValueError("Duplicate reference sample_id")
        seen.add(sid)
        if r.get("geometry_frame") != expected_frame:
            raise ValueError(f"Reference coordinate frame mismatch: {sid}")
        target, valid, ids = load_points(r)
        valid_count = int(valid.sum())
        if not valid_count:
            records.append({"sample_id": sid, "reference_points": 0, "status": "no_reference"})
            continue
        p = pred.get(sid)
        errors = []
        if p:
            if p.get("geometry_frame") != expected_frame:
                raise ValueError(f"Prediction coordinate frame mismatch: {sid}")
            xyz, mask, names = load_points(p)
            if r.get("timestamp_s") is not None:
                if p.get("timestamp_s") is None or abs(p["timestamp_s"] - r["timestamp_s"]) > protocol.get("timestamp_tolerance_s", 0.001):
                    raise ValueError(f"Timestamp mismatch: {sid}")
            lookup = {name: i for i, name in enumerate(names)}
            for j, name in enumerate(ids):
                k = lookup.get(name)
                if valid[j] and k is not None and mask[k]:
                    errors.append(float(np.linalg.norm(xyz[k] - target[j])))
        report = {"sample_id": sid, "subject": identity(r), "reference_points": valid_count,
                  "predicted_valid_points": len(errors), "mean_error_mm": float(np.mean(errors)) if errors else None,
                  "success_points": sum(e <= threshold for e in errors), "status": "scored" if errors else "prediction_failed"}
        subjects[identity(r)].append(report); records.append(report)
    per_subject = []
    for name, rows in sorted(subjects.items()):
        finite = [r["mean_error_mm"] for r in rows if r["mean_error_mm"] is not None]
        total = sum(r["reference_points"] for r in rows)
        per_subject.append({"subject": name, "mean_error_mm": float(np.mean(finite)) if finite else None,
                            "point_coverage": sum(r["predicted_valid_points"] for r in rows) / total,
                            "success_rate_including_failures": sum(r["success_points"] for r in rows) / total,
                            "failed_frames": sum(r["mean_error_mm"] is None for r in rows), "reference_frames": len(rows)})
    means = [r["mean_error_mm"] for r in per_subject if r["mean_error_mm"] is not None]
    return {"protocol": protocol, "aggregation": "mean valid-point error per frame, then equal frames within subject, then equal subjects; coverage reported separately",
            "subject_macro_mean_error_mm": float(np.mean(means)) if means else None,
            "subjects_without_scored_predictions": sum(r["mean_error_mm"] is None for r in per_subject),
            "subject_macro_coverage": float(np.mean([r["point_coverage"] for r in per_subject])) if per_subject else None,
            "subject_macro_success_rate": float(np.mean([r["success_rate_including_failures"] for r in per_subject])) if per_subject else None,
            "per_subject": per_subject, "per_frame": records, "unmatched_predictions": len(set(pred) - seen)}


def trajectory_errors(prediction, reference, timestamps_s, valid):
    """Velocity errors only across adjacent valid observations, with actual dt."""
    p, r, t, v = np.asarray(prediction), np.asarray(reference), np.asarray(timestamps_s), np.asarray(valid, bool)
    if p.shape != r.shape or p.ndim != 3 or p.shape[-1] != 3 or v.shape != p.shape[:2] or t.shape != (len(p),):
        raise ValueError("Expected [T,M,3], mask [T,M], times [T]")
    dt = np.diff(t)
    if not np.isfinite(t).all() or np.any(dt <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    joint = v[1:] & v[:-1]
    delta = np.diff(p - r, axis=0) / dt[:, None, None]
    values = np.linalg.norm(delta, axis=-1)[joint]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite valid trajectory")
    return {"mean_velocity_error_mm_s": float(np.mean(values)) if len(values) else None, "valid_intervals": int(joint.sum())}


def displacement_error(prediction, reference, prediction_neutral, reference_neutral, valid):
    """Subtract separate fixed neutral calibrations; caller supplies a common head frame."""
    p,r=np.asarray(prediction,float),np.asarray(reference,float)
    pn,rn=np.asarray(prediction_neutral,float),np.asarray(reference_neutral,float)
    mask=np.asarray(valid,bool)
    if p.shape!=r.shape or p.ndim!=3 or p.shape[-1]!=3 or pn.shape!=p.shape[1:] or rn.shape!=pn.shape or mask.shape!=p.shape[:2]:
        raise ValueError("Expected trajectories [T,M,3] and fixed neutral arrays [M,3]")
    differences=(p-pn[None])-(r-rn[None])
    values=np.linalg.norm(differences,axis=-1)[mask]
    if not np.isfinite(values).all():raise ValueError("Nonfinite valid displacement")
    return {"mean_displacement_error_mm":float(np.mean(values)) if len(values) else None,"valid_points":int(mask.sum())}
