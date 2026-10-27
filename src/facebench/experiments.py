from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import os
import random
import subprocess
import time
import numpy as np
from PIL import Image
from .data import audit, assert_disjoint, identity
from .io import digest, load_manifest, read_json, read_jsonl, resolve, write_json, write_jsonl

UPSTREAMS = {
    "smirk": {"url": "https://github.com/georgeretsi/smirk.git", "commit": "c7de404c4389f073906a6db1adabf62efcea3f35"},
    "deca": {"url": "https://github.com/yfeng95/DECA.git", "commit": "a11554ae2a2b0f3998cf1fa94dd4db03babb34a2"},
}


def git_info(root):
    def run(*args): return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "tracked_changes": run("status", "--porcelain", "--untracked-files=no")}


def fetch_upstream(model, destination):
    destination = Path(destination).resolve(); source = UPSTREAMS[model]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", source["url"], str(destination)], check=True)
    info = git_info(destination)
    if info["tracked_changes"]: raise ValueError("Upstream has tracked edits; refusing checkout")
    if info["commit"] != source["commit"]:
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["commit"]], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", source["commit"]], check=True)
    return git_info(destination)


def read_config(path):
    config = read_json(path); base = Path(path).resolve().parent
    for key in ("upstream_root", "checkpoint", "train_manifest", "val_manifest", "output_dir"):
        if not config.get(key): raise ValueError(f"Set {key} in the experiment JSON")
        config[key] = str(resolve(base, config[key]))
    return config


def check_experiment(config, check_assets=True):
    if config.get("model") not in UPSTREAMS: raise ValueError("Unknown model")
    for key in ("max_steps", "batch_subjects", "images_per_subject", "save_every", "val_every"):
        if type(config.get(key)) is not int or config[key] < 1: raise ValueError(f"Positive integer {key} required")
    if config["model"] == "smirk" and config["images_per_subject"] != 1:
        raise ValueError("SMIRK adapter samples independent images; images_per_subject must be 1")
    train, val = load_manifest(config["train_manifest"]), load_manifest(config["val_manifest"])
    for rows in (train, val):
        report = audit(rows, check_files=check_assets, require_aux=True)
        if not report["ok"]: raise ValueError(f"Manifest audit failed: {report}")
    assert_disjoint(train, val)
    if {r["dataset_id"] for r in train} != {config["source"]}:
        raise ValueError("Training source does not match group configuration")
    if any(r.get("split") == "test" for r in train + val): raise ValueError("Test split used in training/selection")
    if any(r.get("split") not in (None,"train") for r in train) or any(r.get("split") not in (None,"val") for r in val):
        raise ValueError("Manifest split tags do not match train/validation roles")
    groups = defaultdict(list)
    for r in train: groups[identity(r)].append(r)
    if len(groups) < config["batch_subjects"] or min(map(len, groups.values())) < config["images_per_subject"]:
        raise ValueError("Insufficient subjects/images for the configured same-identity batch")
    expected = config.get("expected_upstream_commit", UPSTREAMS[config["model"]]["commit"])
    upstream = git_info(config["upstream_root"])
    if upstream["commit"] != expected or upstream["tracked_changes"]:
        raise ValueError(f"Upstream differs from reviewed version: {upstream}")
    if check_assets and not Path(config["checkpoint"]).is_file(): raise FileNotFoundError(config["checkpoint"])
    return train, val, upstream


def make_plan(local_path, destination, seeds):
    local, destination = read_json(local_path), Path(destination).resolve()
    base = Path(local_path).resolve().parent
    def locate(value):
        if not value: raise ValueError("Fill all model/source paths in local config before generating runs")
        return str(resolve(base, value))
    results = []
    for model in ("smirk", "deca"):
        for source in ("own", "multiface"):
            for seed in seeds:
                name = f"{model}_{source}_s{seed}"
                config = {"model": model, "source": source, "seed": seed,
                          "upstream_root": locate(local["models"][model]["upstream_root"]),
                          "checkpoint": locate(local["models"][model]["checkpoint"]),
                          "train_manifest": locate(local["sources"][source]["train_manifest"]),
                          "val_manifest": locate(local["val_manifest"]),
                          "output_dir": str(Path(locate(local["output_root"])) / name),
                          "expected_upstream_commit": UPSTREAMS[model]["commit"], "device": local.get("device", "cuda:0"),
                          "max_steps": local.get("max_steps", 100), "batch_subjects": local.get("batch_subjects", 1),
                          "images_per_subject": 1 if model == "smirk" else local.get("deca_images_per_subject", 2),
                          "save_every": local.get("save_every", 50), "val_every": local.get("val_every", 50),
                          "max_val_per_subject": local.get("max_val_per_subject", 8),
                          "learning_rate": local["models"][model].get("learning_rate", 1e-3 if model == "smirk" else 1e-4), "trusted_legacy_assets": local.get("trusted_legacy_assets", False),
                          "upstream_overrides": local["models"][model].get("upstream_overrides", {})}
                write_json(destination / f"{name}.json", config)
                results.append(str(destination / f"{name}.json"))
    write_json(destination / "plan.json", {"purpose": "smoke" if local.get("max_steps",100) <= 100 else "user_configured",
                "training_runs": results, "seeds": seeds, "note": "No automatic claim of sufficient formal training budget"})
    return results


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None: torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def sample_batch(rows_by_person, b, k):
    selected = random.sample(sorted(rows_by_person), b)
    return [r for person in selected for r in random.sample(rows_by_person[person], k)]


def validation_rows(rows, limit):
    if type(limit) is not int or limit < 1: raise ValueError("max_val_per_subject must be positive")
    groups = defaultdict(list)
    for r in rows: groups[identity(r)].append(r)
    result = []
    for person in sorted(groups):
        values = sorted(groups[person], key=lambda x: x["sample_id"])
        indices = np.linspace(0, len(values)-1, min(len(values), limit), dtype=int)
        result.extend(values[i] for i in indices)
    return result


def validate(adapter, rows):
    """Selection metric only: two-dimensional normalized landmark MAE, not 3D accuracy."""
    groups = defaultdict(list)
    for row in rows:
        output = adapter.predict([row])
        errors = (output["landmarks2d"] - output["landmarks_gt"]).abs().mean().item()
        if not np.isfinite(errors): raise FloatingPointError("Nonfinite validation metric")
        groups[(row["dataset_id"], identity(row))].append(errors)
    datasets = defaultdict(list)
    for (dataset, person), values in groups.items(): datasets[dataset].append(float(np.mean(values)))
    means = {key: float(np.mean(values)) for key, values in datasets.items()}
    if not means: raise ValueError("Empty validation set")
    return {"selection_metric": "dataset_macro_2d_landmark_mae_normalized", "score": float(np.mean(list(means.values()))), "per_dataset": means}


def train(config_path, resume=None):
    import torch
    from .models import create_adapter
    config = read_config(config_path)
    rows, val, upstream = check_experiment(config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Nonempty run directory: {output}; use a new directory or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(config[name]) for name in ("train_manifest", "val_manifest", "checkpoint")}
    signature = hashlib.sha256(json.dumps({"config": config, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
    seed_all(config["seed"])
    adapter = create_adapter(config, training=True)
    if hasattr(adapter, "cfg"):
        (output / "effective_upstream_config.txt").write_text(str(adapter.cfg), encoding="utf-8")
    # Upstream imports may enable cudnn benchmarking again.
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    initial_step, best = 0, float("inf")
    if resume:
        saved = torch.load(resume, map_location=config["device"], weights_only=False)
        if saved.get("signature") != signature: raise ValueError("Resume config/data/initial-weight signature mismatch")
        adapter.restore(saved["adapter"]); restore_rng(saved["rng"])
        initial_step, best = saved["step"], saved["best_score"]
    metadata = {"config": config, "hashes": hashes, "upstream": upstream, "signature": signature,
                "adapter_recipe": "upstream SMIRK paths / DECA coarse loss; custom data/schedule; MediaPipe hull mask for initial DECA recipe",
                "torch": torch.__version__, "cuda": torch.version.cuda, "status": "running", "resume_from_step": initial_step}
    write_json(output / "run.json", metadata)
    chosen_val = validation_rows(val, config.get("max_val_per_subject", 8))
    write_jsonl(output / "validation_selection.jsonl", chosen_val)
    groups = defaultdict(list)
    for r in rows: groups[identity(r)].append(r)
    log = output / "metrics.jsonl"
    if resume and log.exists(): write_jsonl(log, (r for r in list(read_jsonl(log)) if r["step"] <= initial_step))
    start = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    def save(step, name):
        packet = {"format": "facebench-v1", "signature": signature, "config": config, "step": step,
                  "best_score": best, "adapter": adapter.state_dict(), "rng": rng_state()}
        tmp = output / (name + ".tmp"); torch.save(packet, tmp); os.replace(tmp, output / name)
    try:
        for step in range(initial_step + 1, config["max_steps"] + 1):
            selected = sample_batch(groups, config["batch_subjects"], config["images_per_subject"])
            losses = adapter.train_step(selected, step - 1)
            record = {"step": step, "losses": losses, "sample_ids": [r["sample_id"] for r in selected]}
            if step % config["val_every"] == 0 or step == config["max_steps"]:
                # Validation must not change the future training random stream.
                state = rng_state()
                try: score = validate(adapter, chosen_val)
                finally: restore_rng(state)
                record["validation"] = score
                if score["score"] < best:
                    best = score["score"]; save(step, "best.pt")
            with log.open("a", encoding="utf-8") as f: f.write(json.dumps(record, allow_nan=False) + "\n")
            if step % config["save_every"] == 0 or step == config["max_steps"]: save(step, "last.pt")
            print(f"step {step}/{config['max_steps']}: {losses}", flush=True)
        metadata.update(status="completed", completed_steps=config["max_steps"], best_validation_score=best,
                        elapsed_s=time.perf_counter()-start,
                        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally: write_json(output / "run.json", metadata)
    return metadata


def infer(config_path, manifest, destination, checkpoint=None):
    from .models import create_adapter
    config = read_config(config_path)
    rows = load_manifest(manifest)
    report = audit(rows, require_aux=True)
    if not report["ok"]: raise ValueError(report)
    if checkpoint: config["checkpoint"] = str(Path(checkpoint).resolve())
    seed_all(config["seed"])
    adapter = create_adapter(config, training=False)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    predictions, failures = [], []
    write_json(destination / "run.json", {"config": config, "manifest_sha256": digest(manifest), "checkpoint_sha256": digest(config["checkpoint"]),
               "upstream": git_info(config["upstream_root"]), "status": "running"})
    for row in rows:
        try:
            result = adapter.predict([row])
            vertices = result["vertices"][0].detach().cpu().numpy()
            if not np.isfinite(vertices).all(): raise FloatingPointError("Nonfinite predicted vertices")
            key = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
            mesh = destination / f"{key}.npz"
            arrays = {"vertices": vertices, "faces": result["faces"].detach().cpu().numpy().reshape(-1,3),
                      "landmarks2d": result["landmarks2d"][0].detach().cpu().numpy()}
            for name, value in result["codes"].items():
                if name != "images" and hasattr(value, "detach"): arrays["code_" + name] = value[0].detach().cpu().numpy()
            np.savez_compressed(mesh, **arrays)
            rgb = result["images"][0].detach().cpu().numpy().transpose(1,2,0)
            rendered = result["rendered"][0].detach().cpu().numpy().transpose(1,2,0)
            Image.fromarray(np.uint8(np.clip(np.concatenate([rgb, rendered, .7*rgb+.3*rendered],axis=1)*255,0,255))).save(destination / f"{key}.jpg")
            predictions.append({"sample_id": row["sample_id"], "dataset_id": row["dataset_id"], "subject_id": row["subject_id"],
                                "identity_id": identity(row),
                                "timestamp_s": row.get("timestamp_s"), "mesh_npz": str(mesh), "model": config["model"],
                                "geometry_unit": "model_native_unverified", "geometry_frame": "model_native"})
        except Exception as exc:
            failures.append({"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"})
        # Append one result instead of rewriting an increasingly large manifest.
        target, entry = ("predictions.jsonl", predictions[-1]) if predictions and predictions[-1]["sample_id"] == row["sample_id"] else ("failures.jsonl", failures[-1])
        with (destination / target).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    for name in ("predictions.jsonl", "failures.jsonl"):
        (destination / name).touch(exist_ok=True)
    result = {"samples": len(rows), "predicted": len(predictions), "failed": len(failures), "status": "completed" if not failures else "completed_with_failures"}
    write_json(destination / "summary.json", result)
    run = read_json(destination / "run.json"); run.update(result); write_json(destination / "run.json", run)
    return result
