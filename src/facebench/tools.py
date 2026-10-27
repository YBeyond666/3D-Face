from __future__ import annotations
import importlib.util
import platform
from pathlib import Path
import subprocess
import sys
import numpy as np
from .data import identity
from .geometry import barycentric_points, transform_points
from .io import read_json, read_jsonl, write_jsonl


def doctor(render_test=False):
    result = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "packages": {m: bool(importlib.util.find_spec(m)) for m in ("torch", "pytorch3d", "cv2", "mediapipe", "face_alignment", "omegaconf", "yacs")}}
    for command in (["nvidia-smi"], ["nvcc", "--version"]):
        try:
            p = subprocess.run(command, text=True, capture_output=True, timeout=20)
            result[command[0]] = {"exit_code": p.returncode, "output": (p.stdout+p.stderr)[-6000:]}
        except (OSError, subprocess.TimeoutExpired) as exc: result[command[0]] = {"error": str(exc)}
    result["cuda_ready"] = False
    if result["packages"]["torch"]:
        import torch
        result.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                value = torch.randn(16,16,device="cuda",requires_grad=True)
                (value @ value.T).square().mean().backward()
                torch.cuda.synchronize()
                result.update(cuda_ready=bool(torch.isfinite(value.grad).all()),
                              gpu_name=torch.cuda.get_device_name(0), gpu_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                              capability=list(torch.cuda.get_device_capability(0)))
            except Exception as exc: result["cuda_error"] = str(exc)
        if render_test:
            try:
                from pytorch3d.structures import Meshes
                from pytorch3d.renderer.mesh import rasterize_meshes
                device="cuda" if torch.cuda.is_available() else "cpu"
                vertices=torch.tensor([[-.7,-.7,1.],[.7,-.7,1.],[0.,.7,1.]],device=device,requires_grad=True)
                faces=torch.tensor([[0,1,2]],device=device)
                _, zbuf, _, _ = rasterize_meshes(Meshes(verts=[vertices],faces=[faces]),image_size=16,faces_per_pixel=1)
                mask=zbuf>=0
                if not mask.any(): raise RuntimeError("Triangle did not rasterize")
                zbuf[mask].sum().backward()
                if vertices.grad is None or not torch.isfinite(vertices.grad).all() or not vertices.grad.abs().sum():
                    raise RuntimeError("Renderer gradient test failed")
                result["renderer"]={"ok":True,"device":device}
            except Exception as exc: result["renderer"]={"ok":False,"error":f"{type(exc).__name__}: {exc}"}
    return result


def export_own_points(rows, output, units, coordinate_frame):
    factors={"mm":1.,"cm":10.,"m":1000.}
    if units not in factors: raise ValueError("unit must be mm, cm or m")
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    caches, result = {}, []
    import hashlib
    for row in rows:
        if row.get("dataset_id")!="own" or not row.get("points_path"): continue
        if row.get("geometry_unit") != units or row.get("geometry_frame") != coordinate_frame:
            raise ValueError("Source points require matching declared unit/frame in manifest")
        path=row["points_path"]
        if path not in caches: caches[path]=list(read_jsonl(path))
        raw=caches[path][row["points_record_index"]]
        ids, xyz, valid=[],[],[]
        for marker in raw.get("markers",[]):
            track=str(marker["id"])
            name=row.get("point_id_mapping",{}).get(track)
            if name is None and track.startswith("F") and track[1:].isdigit(): name=track
            if name is None: raise ValueError(f"Map track ID {track} to a verified physical marker; no automatic F numbering")
            if "xyz_mm" not in marker or units!="mm":
                raise ValueError("Current capture exporter supports raw markers.xyz_mm only; other schemas need an explicit adapter")
            ids.append(name);xyz.append(marker["xyz_mm"]);valid.append(marker.get("valid",True))
        if len(set(ids)) != len(ids): raise ValueError("Two tracks mapped to the same physical marker")
        key=hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
        path=output/f"{key}.npz"
        np.savez_compressed(path,xyz=np.asarray(xyz,dtype=float).reshape(-1,3)*factors[units],valid=np.asarray(valid,bool),point_ids=np.asarray(ids,dtype=str))
        result.append({"sample_id":row["sample_id"],"dataset_id":row["dataset_id"],"subject_id":row["subject_id"],
                       "identity_id":identity(row),"timestamp_s":row.get("timestamp_s"),"points_path":str(path),
                       "geometry_unit":"mm","geometry_frame":coordinate_frame,
                       "reference_type":"raw_capture_markers_not_independently_verified","raw_point_time_ms":raw.get("time_ms")})
    if not result: raise ValueError("No own capture point records found")
    write_jsonl(output/"reference.jsonl",result)
    return {"records":len(result),"manifest":str(output/"reference.jsonl")}


def project_predictions(rows, mapping_path, output):
    """Fixed barycentric correspondences + fixed per-identity calibration only."""
    mapping=read_json(mapping_path)
    if mapping.get("correspondence_reviewed") is not True or mapping.get("calibration_source") is None:
        raise ValueError("Review fixed correspondences and declare calibration source")
    if mapping.get("output_unit")!="mm" or not mapping.get("output_frame"):
        raise ValueError("Mapping must explicitly output mm in a named frame")
    ids=mapping["point_ids"]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate physical marker IDs")
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    import hashlib
    result=[]
    for row in rows:
        if row.get("model") != mapping["model"]: raise ValueError("Wrong model correspondence map")
        with np.load(row["mesh_npz"],allow_pickle=False) as data:
            if len(data["vertices"])!=mapping["vertex_count"]: raise ValueError("Model topology differs")
            points=barycentric_points(data["vertices"],mapping["triangle_vertex_indices"],mapping["barycentric_weights"])
        transforms=mapping["fixed_subject_transforms"]
        key=identity(row)
        if key not in transforms: raise ValueError(f"Missing fixed calibration for {key}")
        points=transform_points(points,transforms[key])
        key=hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
        path=output/f"{key}.npz"
        np.savez_compressed(path,xyz=points,valid=np.isfinite(points).all(1),point_ids=np.asarray(ids,dtype=str))
        result.append({**row,"points_path":str(path),"geometry_unit":"mm","geometry_frame":mapping["output_frame"]})
    write_jsonl(output/"predicted_points.jsonl",result)
    return {"records":len(result),"manifest":str(output/"predicted_points.jsonl")}
