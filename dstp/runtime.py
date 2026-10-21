import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


def atomic_save(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def fingerprint_files(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("dstp", "scripts", "configs") for p in sorted((root / folder).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".sh", ".json"}}


class CosineSchedule:
    """GPS/timm-style non-prefix warmup cosine, with EXPLICIT cooldown length.

    Epoch mode matches timm CosineLRScheduler(t_initial=epochs, warmup_prefix=False).
    Step mode is a continuous interpolation; no implicit batch-size LR scaling.
    """
    def __init__(self, optimizer, args, steps_per_epoch):
        self.optimizer = optimizer
        self.spec = {k: getattr(args, k) for k in (
            "epochs", "cooldown_epochs", "warmup_epochs", "warmup_lr", "min_lr", "lr_interval")}
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.steps_per_epoch = steps_per_epoch
        self.num_updates = 0

    def apply(self, epoch):
        cfg = self.spec
        t = epoch if cfg["lr_interval"] == "epoch" else self.num_updates / self.steps_per_epoch
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if t < cfg["warmup_epochs"]:
                lr = cfg["warmup_lr"] + (base - cfg["warmup_lr"]) * t / cfg["warmup_epochs"]
            else:
                t = min(t, cfg["epochs"])
                lr = cfg["min_lr"] + (base - cfg["min_lr"]) * (1 + math.cos(math.pi * t / cfg["epochs"])) / 2
            group["lr"] = lr
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return dict(spec=self.spec.copy(), base_lrs=self.base_lrs.copy(),
                    steps_per_epoch=self.steps_per_epoch, num_updates=self.num_updates)

    def load_state_dict(self, state):
        if state["spec"] != self.spec or state["steps_per_epoch"] != self.steps_per_epoch:
            raise ValueError("Scheduler config or data length changed on resume")
        if state["base_lrs"] != self.base_lrs:
            raise ValueError("Base learning rates changed on resume")
        self.num_updates = state["num_updates"]


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        # Frozen weights never change: no reason to read/write them every step.
        for target, source in zip(self.model.parameters(), model.parameters()):
            if source.requires_grad:
                target.lerp_(source.detach(), 1 - self.decay)
        for target, source in zip(self.model.buffers(), model.buffers()):
            target.copy_(source)
