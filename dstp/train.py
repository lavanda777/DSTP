"""Single-GPU ViT FGVC / VTAB training baseline. Multi-run scheduling is left to shell."""

import contextlib
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.data import Mixup

from .config import ROOT, parse_args
from .data import FGVC, EpochSampler, make_transform
from .benchmarks import VTAB_CLASSES
from .vtab import VTAB
from .model import build_model
from .baseline import build_optimizer
from .runtime import (CosineSchedule, EMA, atomic_json, atomic_save, fingerprint_files,
                      restore_rng, rng_state, seed_all)


def autocast(args):
    if args.amp == "none":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, args.amp))


def classification_loss(logits, targets, smoothing):
    if targets.ndim == 2:
        # Mixup has already applied label smoothing. Do NOT smooth twice.
        return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
    return F.cross_entropy(logits.float(), targets, label_smoothing=smoothing)


def build_mixup(args, num_classes):
    """Single-view classification augmentation."""
    if not (args.mixup or args.cutmix):
        return None
    return Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                 prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
                 mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=num_classes)


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()
    start = time.perf_counter()
    total_loss = total_correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast(args):
            logits = model(images)
        total_loss += F.cross_entropy(logits.float(), targets, reduction="sum").item()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += targets.numel()
    return dict(loss=total_loss / total, top1=100 * total_correct / total,
                samples=total, seconds=time.perf_counter() - start)


def loader_for(dataset, args, epoch=0, next_batch=0, training=False):
    dataset.epoch = epoch
    sampler = EpochSampler(len(dataset), args.seed, epoch, next_batch * args.batch_size) if training else None
    # Iterator creation/worker base seeds must not advance the model's global RNG.
    generator = torch.Generator().manual_seed(args.seed + epoch)
    batch_size = args.batch_size if training else (args.validation_batch_size or args.batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=args.workers, drop_last=training,
                      pin_memory=args.device.startswith("cuda"), generator=generator,
                      timeout=60 if args.workers else 0)


def check_resume_config(saved, current):
    operational = {"output", "resume", "config", "print_config", "workers", "device",
                   "log_every", "checkpoint_every", "stop_after_steps"}
    differences = {k: (saved.get(k), v) for k, v in current.items()
                   if k not in operational and saved.get(k) != v}
    if differences:
        raise ValueError(f"Training configuration changed on resume: {differences}")


def main(argv=None):
    args = parse_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. This is NOT a reason to silently train on CPU.")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        if args.amp == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16 is unsupported on this GPU")
    resume_path = Path(args.resume).resolve() if args.resume else None
    saved = None
    if resume_path:
        # Validate BEFORE opening a log or writing into the old run directory.
        # Only load checkpoints you trust: full RNG state requires weights_only=False.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("format") != "dstp-baseline-v1":
            raise ValueError("Not a DSTP baseline checkpoint")
        check_resume_config(saved["config"], vars(args))
    output = Path(args.output).resolve() if args.output else (
        resume_path.parent if resume_path else ROOT / "result" / ("vtab" if args.dataset in VTAB_CLASSES else "") /
        f"{args.dataset}_adamw_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if resume_path is not None and output != resume_path.parent:
        raise ValueError("Resume in the checkpoint's own run directory; refusing to overwrite another run")
    if output.exists() and any(output.iterdir()) and resume_path is None:
        raise FileExistsError(f"Refusing to overwrite a nonempty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "train.log")], force=True)
    log = logging.getLogger("dstp")
    log.info("ViT AdamW baseline | %s | output=%s", args.dataset, output)
    log.info("Epoch budget: %d scheduled + %d cooldown = %d total; no hidden extension",
             args.epochs, args.cooldown_epochs, args.total_epochs)
    seed_all(args.seed, args.deterministic)
    dataset_class = VTAB if args.dataset in VTAB_CLASSES else FGVC
    train_data = dataset_class(args.data_root, args.dataset, args.train_split,
                      make_transform(True, args.augmentation), args.seed, split_root=args.split_root)
    eval_data = dataset_class(args.data_root, args.dataset, args.eval_split,
                             make_transform(False, "vtab" if args.augmentation == "vtab" else "simple"),
                             split_root=args.split_root)
    steps_per_epoch = len(train_data) // args.batch_size
    if steps_per_epoch < 1:
        raise ValueError("Training split has fewer images than one full batch")
    model = build_model(args.pretrained, train_data.num_classes, args.drop_path).to(device)
    optimizer, manifest = build_optimizer(model, args)
    scheduler = CosineSchedule(optimizer, args, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "float16")
    ema = EMA(model, args.ema_decay) if args.ema_decay else None
    mixup = build_mixup(args, train_data.num_classes)
    log.info("Training recipe: single-view %s; mixup_alpha=%g cutmix_alpha=%g prob=%g "
             "cutmix_switch_prob=%g mode=%s smoothing=%g drop_path=%g AMP=%s EMA_decay=%g; "
             "loss=%s; no contrastive training loss",
             args.augmentation, args.mixup, args.cutmix, args.mixup_prob, args.mixup_switch_prob,
             args.mixup_mode, args.smoothing, args.drop_path, args.amp, args.ema_decay,
             "soft-target CE (smoothing applied once by Mixup)" if mixup else
             "label-smoothed CE" if args.smoothing else "hard-label CE")
    progress = dict(epoch=0, next_batch=0, batches_seen=0, loss_sum=0.0, samples=0,
                    best_test_top1=None, best_epoch=None)
    timing = dict(training_seconds=0.0, forward_seconds=0.0, backward_seconds=0.0,
                  optimizer_seconds=0.0, data_wait_seconds=0.0, evaluation_seconds=0.0,
                  checkpoint_seconds=0.0, timed_steps=0, amp_skipped_steps=0, peak_memory_bytes=0)
    history = []
    weights_hash = hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest()
    environment = dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__,
        numpy=np.__version__, cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pretrained_sha256=weights_hash, source_hashes=fingerprint_files(ROOT))
    if resume_path:
        if saved["environment"]["source_hashes"] != environment["source_hashes"]:
            log.warning("Source files changed since checkpoint; see recorded source hashes before claiming exact reproduction")
        if saved["data_manifest"] != train_data.manifest or saved["environment"]["pretrained_sha256"] != weights_hash:
            raise ValueError("Data splits or pretrained weights changed on resume")
        if saved.get("eval_manifest") != eval_data.manifest:
            raise ValueError("Evaluation/test data changed on resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        if ema:
            ema.model.load_state_dict(saved["ema"])
        progress, timing, history = saved["progress"], saved["timing"], saved["history"]
        restore_rng(saved["rng"])
        del saved
    else:
        atomic_json(output / "config.json", vars(args))
        atomic_json(output / "parameter_manifest.json", manifest)
        atomic_json(output / "data_manifest.json", train_data.manifest)
        atomic_json(output / "test_manifest.json", eval_data.manifest)
        atomic_json(output / "environment.json", environment)
    with (output / "commands.jsonl").open("a") as f:
        f.write(json.dumps(dict(argv=sys.argv, resolved_config=vars(args), cwd=os.getcwd()), ensure_ascii=False) + "\n")
    log.info("Evaluation workflow: train+val=%d; test=%d every epoch; steps/epoch=%d (drop_last=True); "
             "best.pt selected by test Top-1 (%s); ties keep the earlier checkpoint",
             len(train_data), len(eval_data), steps_per_epoch, "EMA" if ema else "raw")
    log.info("Test is evaluation-only (no backprop), but is used for checkpoint selection; "
             "this reports best-test, not an untouched-test estimate. Hyperparameters are fixed before training.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(filename="last.pt"):
        started = time.perf_counter()
        payload = dict(format="dstp-baseline-v1", model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
            ema=ema.model.state_dict() if ema else None, config=vars(args), progress=progress.copy(),
            timing=timing.copy(), history=history, data_manifest=train_data.manifest,
            eval_manifest=eval_data.manifest, environment=environment)
        atomic_save(output / filename, payload)
        timing["checkpoint_seconds"] += time.perf_counter() - started
        atomic_json(output / "progress.json", {**progress, "optimizer_updates": scheduler.num_updates,
                   "total_epochs": args.total_epochs, "complete": progress["epoch"] >= args.total_epochs})

    for epoch in range(progress["epoch"], args.total_epochs):
        model.train()
        next_batch = progress["next_batch"]
        loader = loader_for(train_data, args, epoch, next_batch, training=True)
        previous_end = time.perf_counter()
        epoch_training_start = timing["training_seconds"]
        for batch_index, (images, targets) in enumerate(loader, start=next_batch):
            timing["data_wait_seconds"] += time.perf_counter() - previous_end
            started = time.perf_counter()
            lr_values = scheduler.apply(epoch)
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if mixup:
                images, targets = mixup(images, targets)
            optimizer.zero_grad(set_to_none=True)
            cpu_marks = [time.perf_counter()]
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if device.type == "cuda" else None
            if events:
                events[0].record()
            with autocast(args):
                logits = model(images)
                loss = classification_loss(logits, targets, args.smoothing)
            if events:
                events[1].record()
            cpu_marks.append(time.perf_counter())
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError("Nonfinite loss; no optimizer step or checkpoint written")
            scaler.scale(loss).backward()
            if events:
                events[2].record()
            cpu_marks.append(time.perf_counter())
            before_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if events:
                events[3].record()
            cpu_marks.append(time.perf_counter())
            updated = scaler.get_scale() >= before_scale
            if updated:
                scheduler.num_updates += 1
                if ema:
                    ema.update(model)
            else:
                timing["amp_skipped_steps"] += 1
            if events:
                torch.cuda.synchronize(device)
            loss_value = loss.detach().item()
            if not np.isfinite(loss_value):
                raise FloatingPointError("Nonfinite loss. Last good checkpoint is retained; no bad state saved.")
            elapsed = time.perf_counter() - started
            components = ([events[i].elapsed_time(events[i + 1]) / 1000 for i in range(3)] if events
                          else [cpu_marks[i + 1] - cpu_marks[i] for i in range(3)])
            for key, value in zip(("forward_seconds", "backward_seconds", "optimizer_seconds"), components):
                timing[key] += value
            timing["training_seconds"] += elapsed
            timing["timed_steps"] += 1
            if device.type == "cuda":
                timing["peak_memory_bytes"] = max(timing["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
            progress["next_batch"] = batch_index + 1
            progress["batches_seen"] += 1
            progress["loss_sum"] += loss_value * images.shape[0]
            progress["samples"] += images.shape[0]
            row = dict(epoch=epoch, batch=batch_index, update=scheduler.num_updates, loss=loss_value,
                lr=lr_values, time_step_seconds=elapsed, forward_seconds=components[0],
                backward_seconds=components[1], optimizer_seconds=components[2], updated=updated,
                peak_memory_bytes=timing["peak_memory_bytes"])
            with (output / "steps.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            if batch_index % args.log_every == 0:
                log.info("epoch=%d/%d batch=%d/%d update=%d loss=%.5f lr=%s time/step=%.4fs",
                         epoch + 1, args.total_epochs, batch_index + 1, steps_per_epoch,
                         scheduler.num_updates, loss_value, lr_values, elapsed)
            if args.stop_after_steps and scheduler.num_updates >= args.stop_after_steps:
                checkpoint()
                atomic_json(output / "summary.json", dict(status="paused", progress=progress, timing=timing))
                log.info("Graceful stop requested. Resume with the same script + --resume %s", output / "last.pt")
                return
            if args.checkpoint_every and progress["batches_seen"] % args.checkpoint_every == 0:
                checkpoint()
            previous_end = time.perf_counter()
        row = dict(epoch=epoch + 1, updates=scheduler.num_updates,
                   train_loss=progress["loss_sum"] / max(1, progress["samples"]),
                   training_seconds_this_process_epoch=timing["training_seconds"] - epoch_training_start)
        eval_loader = loader_for(eval_data, args)
        row["test"] = evaluate(model, eval_loader, args, device)
        timing["evaluation_seconds"] += row["test"]["seconds"]
        if ema:
            row["test_ema"] = evaluate(ema.model, eval_loader, args, device)
            timing["evaluation_seconds"] += row["test_ema"]["seconds"]
        metric = row["test_ema" if ema else "test"]["top1"]
        is_best = progress["best_test_top1"] is None or metric > progress["best_test_top1"]
        if is_best:
            progress["best_test_top1"], progress["best_epoch"] = metric, epoch + 1
        history.append(row)
        log.info("epoch_result=%s", json.dumps(row))
        progress.update(epoch=epoch + 1, next_batch=0, loss_sum=0.0, samples=0)
        if is_best:
            checkpoint("best.pt")
        checkpoint()
        atomic_json(output / "epochs.json", history)
    best_row = next((row for row in history if row["epoch"] == progress["best_epoch"]), None)
    summary = dict(status="complete", updates=scheduler.num_updates,
        train_split=args.train_split, eval_split=args.eval_split, eval_interval=args.eval_interval,
        checkpoint_metric="test_top1", epochs=args.total_epochs,
        best_epoch=progress["best_epoch"], best_test_top1=progress["best_test_top1"],
        best_checkpoint=str(output / "best.pt"), last_checkpoint=str(output / "last.pt"),
        best_epoch_metrics=best_row,
        timing=timing, time_per_step_seconds=timing["training_seconds"] / max(1, timing["timed_steps"]),
        time_per_epoch_training_seconds=timing["training_seconds"] / args.total_epochs,
        time_per_epoch_with_data_seconds=(timing["training_seconds"] + timing["data_wait_seconds"]) / args.total_epochs,
        peak_gpu_memory_gib=timing["peak_memory_bytes"] / 2 ** 30,
        final_epoch=history[-1] if history else None,
        primary_model="ema" if ema else "raw",
        method="adamw_full_finetune_baseline")
    atomic_json(output / "summary.json", summary)
    log.info("Finished: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
