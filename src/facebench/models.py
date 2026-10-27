"""Adapters call upstream implementations; no upstream source is redistributed."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import numpy as np
from .data import read_rgb
from .preprocessing import crop_matrix, apply_crop, normalized_landmarks


@contextmanager
def upstream_context(root, legacy_load=False):
    """Upstreams use relative assets. Scope their cwd and trusted legacy loads."""
    import torch
    old_cwd, old_path, old_load = Path.cwd(), list(sys.path), torch.load
    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old_load(*args, **kwargs)
    try:
        os.chdir(root); sys.path.insert(0, str(root))
        if legacy_load: torch.load = trusted_load
        yield
    finally:
        torch.load = old_load
        sys.path[:] = old_path; os.chdir(old_cwd)


def auxiliary(row):
    if not row.get("aux_path"):
        raise ValueError(f'Preprocess first: {row["sample_id"]}')
    with np.load(row["aux_path"], allow_pickle=False) as data:
        result = {key: data[key] for key in ("fan", "mediapipe", "face_mask")}
    if result["fan"].shape != (68, 2) or result["mediapipe"].shape[0] < 468:
        raise ValueError(f'Invalid landmark cache: {row["sample_id"]}')
    return result


class ModelAdapter:
    def __init__(self, config, training=False):
        import torch
        self.torch, self.config, self.training = torch, config, training
        self.root = Path(config["upstream_root"]).resolve()
        self.checkpoint = Path(config["checkpoint"]).resolve()
        self.device = config.get("device", "cuda:0")
        self.legacy = config.get("trusted_legacy_assets", False)
        if not self.root.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError("upstream_root and checkpoint must exist")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU training fallback")
        with self.context(): self.initialize()

    def context(self): return upstream_context(self.root, self.legacy)

    def collate(self, rows, training):
        with self.context():
            items = [self.prepare(row, training) for row in rows]
        batch = self.torch.utils.data.default_collate(items)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _guard_gradients(self, parameters):
        def guard(grad):
            if not self.torch.isfinite(grad).all():
                raise FloatingPointError("Nonfinite gradient; optimizer update aborted")
            return grad
        for p in parameters:
            if p.requires_grad: p.register_hook(guard)


class SmirkAdapter(ModelAdapter):
    def initialize(self):
        from omegaconf import OmegaConf
        self.cfg = OmegaConf.load(str(self.root / "configs/config_train.yaml"))
        overrides = self.config.get("upstream_overrides", {})
        self.cfg = OmegaConf.merge(self.cfg, OmegaConf.create(overrides))
        self.cfg.device = self.device
        self.cfg.train.lr = self.config.get("learning_rate", self.cfg.train.lr)
        self.cfg.train.visualize_every = max(1, self.cfg.train.visualize_every)
        self.cfg.train.log_losses_every = 0
        from src.smirk_encoder import SmirkEncoder
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
        state = self.torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        packet = state.get("adapter") if state.get("format") == "facebench-v1" else None
        if self.training:
            from src.smirk_trainer import SmirkTrainer
            if not self.cfg.arch.enable_fuse_generator or self.cfg.train.loss_weights.cycle_loss <= 0:
                raise ValueError("Initial SMIRK training adapter requires reconstruction + cycle paths")
            self.trainer = SmirkTrainer(self.cfg).to(self.device)
            # Detect incomplete encoder/generator checkpoints instead of random fallback.
            initial = packet["model"] if packet else state
            expected = {k for k in self.trainer.state_dict() if k.startswith(("smirk_encoder.", "smirk_generator."))}
            if not expected.issubset(initial): raise ValueError("SMIRK encoder/generator checkpoint is incomplete")
            self.trainer.load_state_dict(initial, strict=False)
            self.trainer.create_base_encoder()
            for p in self.trainer.base_encoder.parameters(): p.requires_grad_(False)
            for branch in ("pose", "shape", "expression"):
                if not self.cfg.train.get("optimize_" + branch, False):
                    for p in getattr(self.trainer.smirk_encoder, branch + "_encoder").parameters(): p.requires_grad_(False)
            original_train = self.trainer.train
            def train_mode():
                original_train()
                # Keep frozen branches' BatchNorm buffers fixed as well as their weights.
                for branch in ("pose", "shape", "expression"):
                    if not self.cfg.train.get("optimize_" + branch, False):
                        getattr(self.trainer.smirk_encoder, branch + "_encoder").eval()
            self.trainer.train = train_mode
            self.trainer.configure_optimizers(self.config["max_steps"])
            self.encoder, self.flame, self.renderer = self.trainer.smirk_encoder, self.trainer.flame, self.trainer.renderer
            self._guard_gradients(self.trainer.parameters())
            self.last_losses = {}
            self.trainer.logging = lambda index, losses, phase: self.last_losses.update(losses)
            # Check each upstream path loss BEFORE its internal optimizer update.
            for method_name, loss_index in (("step1", 2), ("step2", 2)):
                original = getattr(self.trainer, method_name)
                def guarded(*args, _original=original, _index=loss_index, **kwargs):
                    result = _original(*args, **kwargs)
                    if not self.torch.isfinite(result[_index]).all():
                        raise FloatingPointError("Nonfinite SMIRK path loss; update aborted")
                    return result
                setattr(self.trainer, method_name, guarded)
        else:
            self.encoder = SmirkEncoder(n_exp=self.cfg.arch.num_expression, n_shape=self.cfg.arch.num_shape).to(self.device)
            initial = packet["model"] if packet else state
            weights = {k.removeprefix("smirk_encoder."): v for k, v in initial.items() if k.startswith("smirk_encoder.")}
            self.encoder.load_state_dict(weights, strict=True)
            self.flame = FLAME(n_exp=self.cfg.arch.num_expression, n_shape=self.cfg.arch.num_shape).to(self.device)
            self.renderer = Renderer(render_full_head=False).to(self.device)
        from datasets.base_dataset import BaseDataset
        self.train_preparer = BaseDataset([], self.cfg, test=False)
        self.eval_preparer = BaseDataset([], self.cfg, test=True)

    def prepare(self, row, training):
        a, image = auxiliary(row), read_rgb(row)
        preparer = self.train_preparer if training else self.eval_preparer
        return preparer.prepare_data(image[..., ::-1].copy(), a["fan"], a["mediapipe"])

    def train_step(self, rows, step):
        batch = self.collate(rows, True)
        with self.context():
            self.trainer.set_freeze_status(self.cfg, step, 0)
            self.last_losses = {}
            self.trainer.step(batch, step, phase="train")
        if not all(np.isfinite(float(x)) for x in self.last_losses.values()):
            raise FloatingPointError("Nonfinite SMIRK loss")
        return {k: float(v) for k, v in self.last_losses.items()}

    def predict(self, rows):
        batch = self.collate(rows, False)
        with self.context(), self.torch.no_grad():
            self.encoder.eval()
            codes = self.encoder(batch["img"])
            geometry = self.flame(codes)
            rendered = self.renderer(geometry["vertices"], codes["cam"], landmarks_fan=geometry["landmarks_fan"], landmarks_mp=geometry["landmarks_mp"])
            return {"vertices": geometry["vertices"], "landmarks2d": rendered["landmarks_fan"],
                    "landmarks_gt": batch["landmarks_fan"], "images": batch["img"], "rendered": rendered["rendered_img"],
                    "faces": self.flame.faces_tensor, "codes": codes}

    def state_dict(self):
        t = self.trainer
        return {"model": {k: v for k, v in t.state_dict().items() if k.startswith(("smirk_encoder.", "smirk_generator.", "base_encoder."))},
                "encoder_optimizer": t.encoder_optimizer.state_dict(), "generator_optimizer": t.smirk_generator_optimizer.state_dict(),
                "encoder_scheduler": t.encoder_scheduler.state_dict(), "generator_scheduler": t.smirk_generator_scheduler.state_dict()}

    def restore(self, state):
        t = self.trainer
        t.load_state_dict(state["model"], strict=False)
        t.encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        t.smirk_generator_optimizer.load_state_dict(state["generator_optimizer"])
        t.encoder_scheduler.load_state_dict(state["encoder_scheduler"])
        t.smirk_generator_scheduler.load_state_dict(state["generator_scheduler"])


class DecaAdapter(ModelAdapter):
    def initialize(self):
        from decalib.utils.config import cfg
        self.cfg = cfg.clone()
        self.cfg.output_dir = self.config["output_dir"]
        self.cfg.pretrained_modelpath = str(self.checkpoint)
        self.cfg.device = self.device
        self.cfg.rasterizer_type = "pytorch3d"
        self.cfg.dataset.batch_size = self.config.get("batch_subjects", 1)
        self.cfg.dataset.K = self.config.get("images_per_subject", 2)
        self.cfg.train.resume = False
        self.cfg.train.write_summary = False
        self.cfg.train.train_detail = False
        self.cfg.train.lr = self.config.get("learning_rate", 1e-4)
        def update(node, changes):
            for key, value in changes.items():
                if key not in node: raise ValueError(f"Unknown upstream DECA key: {key}")
                if isinstance(value, dict): update(node[key], value)
                else: node[key] = value
        update(self.cfg, self.config.get("upstream_overrides", {}))
        if self.cfg.dataset.batch_size != self.config.get("batch_subjects",1) or self.cfg.dataset.K != self.config.get("images_per_subject",2):
            raise ValueError("Set batch grouping in the outer experiment config, not upstream overrides")
        if self.cfg.train.train_detail:
            raise ValueError("v0.1 implements coarse training; detail requires a separate validated protocol")
        Path(self.cfg.output_dir, self.cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
        # Our checkpoints wrap native model weights. DECA constructor expects native keys.
        saved = self.torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        packet = saved.get("adapter") if saved.get("format") == "facebench-v1" else None
        native = packet["model"] if packet else saved
        if not all(key in native for key in ("E_flame", "E_detail", "D_detail")):
            raise ValueError("DECA checkpoint lacks native component weights")
        from decalib.deca import DECA
        # Suppress constructor checkpoint loading only for our packet; load strictly below.
        if packet: self.cfg.pretrained_modelpath = "__facebench_explicit_load__"
        self.deca = DECA(config=self.cfg, device=self.device)
        for key in ("E_flame", "E_detail", "D_detail"):
            getattr(self.deca, key).load_state_dict(native[key], strict=True)
        self.cfg.pretrained_modelpath = str(self.checkpoint)
        if self.training:
            from decalib.trainer import Trainer
            # Constructor load_checkpoint is replaced only on our subclass, not upstream source.
            class NativeWeightsAlreadyLoaded(Trainer):
                def load_checkpoint(instance): instance.global_step = 0
            self.trainer = NativeWeightsAlreadyLoaded(self.deca, config=self.cfg, device=self.device)
            self._guard_gradients(self.deca.E_flame.parameters())

    def prepare(self, row, training):
        import random
        a, image = auxiliary(row), read_rgb(row)
        if a["face_mask"].shape != image.shape[:2]: raise ValueError("Mask/image dimensions differ")
        scale = random.uniform(self.cfg.dataset.scale_min, self.cfg.dataset.scale_max) if training else (self.cfg.dataset.scale_min + self.cfg.dataset.scale_max)/2
        matrix = crop_matrix(a["fan"], scale, self.cfg.dataset.image_size)
        image = apply_crop(image, matrix, self.cfg.dataset.image_size).astype(np.float32) / 255
        mask = apply_crop(a["face_mask"] * 255, matrix, self.cfg.dataset.image_size, mask=True).astype(np.float32)/255
        return {"image": self.torch.from_numpy(image.transpose(2, 0, 1).copy()),
                "landmark": self.torch.from_numpy(normalized_landmarks(a["fan"], matrix, self.cfg.dataset.image_size)),
                "mask": self.torch.from_numpy(mask.copy())}

    def train_step(self, rows, step):
        batch = self.collate(rows, True)
        b, k = self.cfg.dataset.batch_size, self.cfg.dataset.K
        batch = {name: value.reshape(b, k, *value.shape[1:]) for name, value in batch.items()}
        with self.context():
            losses, _ = self.trainer.training_step(batch, step)
            loss = losses["all_loss"]
            if not self.torch.isfinite(loss): raise FloatingPointError("Nonfinite DECA loss")
            self.trainer.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.trainer.opt.step()
        return {key: float(value.detach()) for key, value in losses.items()}

    def predict(self, rows):
        batch = self.collate(rows, False)
        with self.context(), self.torch.no_grad():
            self.deca.eval()
            codes = self.deca.encode(batch["image"], use_detail=False)
            result = self.deca.decode(codes, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
            return {"vertices": result["verts"], "landmarks2d": result["landmarks2d"][..., :2], "landmarks_gt": batch["landmark"][..., :2],
                    "images": batch["image"], "rendered": result["rendered_images"], "faces": self.deca.flame.faces_tensor, "codes": codes}

    def state_dict(self):
        return {"model": {key: getattr(self.deca, key).state_dict() for key in ("E_flame", "E_detail", "D_detail")},
                "optimizer": self.trainer.opt.state_dict()}

    def restore(self, state):
        for key, value in state["model"].items(): getattr(self.deca, key).load_state_dict(value, strict=True)
        self.trainer.opt.load_state_dict(state["optimizer"])


def create_adapter(config, training=False):
    if config["model"] not in {"smirk", "deca"}: raise ValueError("model must be smirk or deca")
    return {"smirk": SmirkAdapter, "deca": DecaAdapter}[config["model"]](config, training)
