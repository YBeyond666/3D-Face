from __future__ import annotations
import numpy as np


def transform_points(points, matrix):
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if points.shape[-1] != 3 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected (...,3) points and finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Only affine transforms supported")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def barycentric_points(vertices, triangles, weights):
    vertices, triangles, weights = np.asarray(vertices), np.asarray(triangles), np.asarray(weights, float)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or weights.shape != triangles.shape:
        raise ValueError("triangles and weights must have shape [markers,3]")
    if not np.issubdtype(triangles.dtype, np.integer) or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Triangle vertex indices out of range")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(1), 1):
        raise ValueError("Barycentric weights must be finite, nonnegative and sum to one")
    return (vertices[triangles] * weights[..., None]).sum(axis=1)


def krt_read(path):
    """Read Multiface KRT blocks: camera, K(3), distortion(1), Rt(3)."""
    from pathlib import Path
    lines = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    result, i = {}, 0
    while i < len(lines):
        if i + 8 > len(lines):
            raise ValueError("Truncated KRT block")
        camera = lines[i]
        k = np.array([[float(x) for x in s.split()] for s in lines[i + 1:i + 4]])
        distortion = np.array([float(x) for x in lines[i + 4].split()])
        rt = np.array([[float(x) for x in s.split()] for s in lines[i + 5:i + 8]])
        if camera in result or k.shape != (3, 3) or rt.shape != (3, 4):
            raise ValueError("Invalid or duplicate KRT camera")
        result[camera] = {"intrinsics": k.tolist(), "distortion": distortion.tolist(), "extrinsics_raw": rt.tolist()}
        i += 8
    return result


def obj_vertices(path):
    from pathlib import Path
    vertices = []
    with Path(path).open() as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No OBJ vertices: {path}")
    return np.asarray(vertices, np.float32)
