from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import numpy as np
from PIL import Image

from .io import load_manifest, read_json, read_jsonl, resolve


def identity(row):
    return row.get("identity_id") or f'{row["dataset_id"]}:{row["subject_id"]}'


def audit(rows, check_files=True, require_aux=False):
    errors, seen, subjects, physical = [], set(), set(), set()
    for i, r in enumerate(rows):
        prefix = f"row {i + 1} ({r.get('sample_id', '?')})"
        for key in ("sample_id", "dataset_id", "subject_id", "sequence_id", "camera_id"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                errors.append(f"{prefix}: missing/non-string {key}")
        sid = r.get("sample_id")
        if sid in seen:
            errors.append(f"{prefix}: duplicate sample_id")
        seen.add(sid)
        if r.get("dataset_id") and r.get("subject_id"):
            subjects.add(identity(r))
        has_image, has_video = bool(r.get("image_path")), bool(r.get("video_path"))
        if has_image == has_video:
            errors.append(f"{prefix}: provide exactly one image_path or video_path")
        if has_video and (not isinstance(r.get("frame_index"), int) or r["frame_index"] < 0):
            errors.append(f"{prefix}: video frame_index must be a nonnegative integer")
        source = (r.get("image_path") or r.get("video_path"), r.get("frame_index") if has_video else None)
        if source in physical:
            errors.append(f"{prefix}: repeated image/video frame")
        physical.add(source)
        if r.get("timestamp_s") is not None and not np.isfinite(r["timestamp_s"]):
            errors.append(f"{prefix}: nonfinite timestamp")
        if require_aux and not r.get("aux_path"):
            errors.append(f"{prefix}: preprocessing aux_path required")
        if check_files:
            for key in ("image_path", "video_path", "aux_path", "mesh_path", "headpose_path", "calibration_path", "points_path"):
                if r.get(key) and not Path(r[key]).is_file():
                    errors.append(f"{prefix}: missing {key}: {r[key]}")
    return {"samples": len(rows), "subjects": len(subjects), "errors": errors,
            "recorded_timestamp_samples": sum(r.get("timestamp_s") is not None for r in rows),
            "point_reference_samples": sum(bool(r.get("points_path")) for r in rows),
            "mesh_reference_samples": sum(bool(r.get("mesh_path")) for r in rows),
            "ok": bool(rows) and not errors}


def assert_disjoint(*groups):
    ids, samples, sources = {}, {}, {}
    for group_index, rows in enumerate(groups):
        result = audit(rows, check_files=False)
        if not result["ok"]:
            raise ValueError(f"Invalid split {group_index}: {result['errors'][:5]}")
        for r in rows:
            keys = [(ids, identity(r)), (samples, r["sample_id"]),
                    (sources, (r.get("image_path") or r.get("video_path"), r.get("frame_index")))]
            for registry, key in keys:
                if key in registry and registry[key] != group_index:
                    raise ValueError(f"Cross-split leakage: {key}")
                registry[key] = group_index


def split_subjects(rows, counts, seed=20261027):
    """Exactly allocate identities. Never fall back to frame splitting."""
    if set(counts) != {"train", "val", "test"} or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("train/val/test counts must each be positive integers")
    people = sorted({identity(r) for r in rows})
    if sum(counts.values()) != len(people):
        raise ValueError(f"Counts sum to {sum(counts.values())}; dataset has {len(people)} identities")
    random.Random(seed).shuffle(people)
    assignment, start = {}, 0
    for name in ("train", "val", "test"):
        for person in people[start:start + counts[name]]:
            assignment[person] = name
        start += counts[name]
    result = {name: [dict(r, split=name) for r in rows if assignment[identity(r)] == name] for name in counts}
    assert_disjoint(*result.values())
    return result


def balanced_subset(rows, subjects, frames_per_subject, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[identity(r)].append(r)
    eligible = sorted(p for p, frames in groups.items() if len(frames) >= frames_per_subject)
    if subjects < 1 or frames_per_subject < 1 or len(eligible) < subjects:
        raise ValueError("Insufficient identities/frames for requested matched budget; no oversampling is hidden")
    rng, selected = random.Random(seed), []
    for person in rng.sample(eligible, subjects):
        selected.extend(rng.sample(sorted(groups[person], key=lambda r: r["sample_id"]), frames_per_subject))
    return sorted(selected, key=lambda r: r["sample_id"])


def multiface_records(root, cameras=None, sequences=None, frame_step=1, require_mesh=False, geometry_unit=None):
    if frame_step < 1:
        raise ValueError("frame_step must be >=1")
    root = Path(root).resolve()
    for images in sorted(root.rglob("images")):
        if not images.is_dir() or not (images.parent / "KRT").is_file():
            continue
        person_root = images.parent
        subject = person_root.name
        for sequence in sorted(images.iterdir()):
            if not sequence.is_dir() or (sequences and sequence.name not in sequences):
                continue
            for camera in sorted(sequence.iterdir()):
                if not camera.is_dir() or (cameras and camera.name not in cameras):
                    continue
                for image in sorted(camera.glob("*.png"))[::frame_step]:
                    frame = image.stem
                    mesh = person_root / "tracked_mesh" / sequence.name / f"{frame}.obj"
                    pose = person_root / "tracked_mesh" / sequence.name / f"{frame}_transform.txt"
                    if require_mesh and not mesh.is_file():
                        raise ValueError(f"Missing mesh: {mesh}")
                    yield {"schema_version": 1, "sample_id": f"multiface/{subject}/{sequence.name}/{camera.name}/{frame}",
                           "dataset_id": "multiface", "subject_id": subject, "session_id": subject,
                           "sequence_id": sequence.name, "camera_id": camera.name,
                           "image_path": str(image), "source_frame_id": frame,
                           "timestamp_s": None, "timestamp_source": "not_inferred_from_filename",
                           "mesh_path": str(mesh) if mesh.is_file() else None,
                           "headpose_path": str(pose) if pose.is_file() else None,
                           "calibration_path": str(person_root / "KRT"),
                           "geometry_unit": geometry_unit, "geometry_frame": "upstream_requires_verification"}


def dotted(value, key):
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def own_records(spec_path):
    """Explicit session specification; no guessed file naming or clock domains."""
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    for session in spec["sessions"]:
        if type(session.get("frame_step",1)) is not int or session.get("frame_step",1) < 1:
            raise ValueError("frame_step must be a positive integer")
        base = spec_path.parent
        video = resolve(base, session["video_path"])
        times = None
        if session.get("timestamps_path"):
            stamps = list(read_jsonl(resolve(base, session["timestamps_path"])))
            if session.get("timestamp_stream"):
                stamps = [r for r in stamps if r.get("stream") == session["timestamp_stream"]]
            times = {}
            for j, r in enumerate(stamps):
                index = dotted(r, session["timestamp_index_key"]) if session.get("timestamp_index_key") else j
                if index in times:
                    raise ValueError(f"Duplicate timestamp frame {index}")
                times[index] = float(dotted(r, session["timestamp_value_key"])) * session["timestamp_to_seconds"]
            if any(b <= a for a, b in zip(times.values(), list(times.values())[1:])):
                raise ValueError("Timestamps must be strictly increasing in recorded order")
            if not times:
                raise ValueError("Timestamp file/stream selection is empty; verify actual capture fields")
        points, point_path = {}, None
        if session.get("points_path"):
            point_path = resolve(base, session["points_path"])
            if not session.get("points_rgb_index_key"):
                raise ValueError("points_rgb_index_key required; point/RGB order cannot be guessed")
            for line, r in enumerate(read_jsonl(point_path)):
                index = dotted(r, session["points_rgb_index_key"])
                if index is not None:
                    if index in points:
                        raise ValueError(f"More than one point record maps to RGB frame {index}; define a selection policy first")
                    points[index] = line
        count = session.get("frame_count")
        if count is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Set frame_count or install the video extra to inspect MP4") from exc
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError(f"Cannot open {video}")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if count <= 0:
            raise ValueError("Empty video")
        for i in range(0, count, session.get("frame_step", 1)):
            r = {"schema_version": 1, "dataset_id": "own", "subject_id": session["subject_id"],
                 "session_id": session["session_id"], "sequence_id": session["sequence_id"], "camera_id": session["camera_id"],
                 "video_path": str(video), "frame_index": i, "timestamp_s": times.get(i) if times else None,
                 "timestamp_source": "recorded" if times and i in times else "unavailable",
                 "clock_domain": session.get("clock_domain"), "geometry_unit": session.get("points_unit"),
                 "geometry_frame": session.get("coordinate_frame"), "point_id_mapping": session.get("point_id_mapping", {}),
                 "points_path": str(point_path) if i in points else None, "points_record_index": points.get(i),
                 "calibration_path": str(resolve(base, session["calibration_path"])) if session.get("calibration_path") else None}
            r["sample_id"] = f'own/{r["subject_id"]}/{r["session_id"]}/{r["sequence_id"]}/{r["camera_id"]}/{i:08d}'
            if session.get("identity_id"):
                r["identity_id"] = session["identity_id"]
            yield r


def read_rgb(row):
    if row.get("image_path"):
        with Image.open(row["image_path"]) as im:
            return np.asarray(im.convert("RGB"))
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MP4 input requires pip install -e '.[video]'") from exc
    cap = cv2.VideoCapture(row["video_path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode {row["sample_id"]}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
