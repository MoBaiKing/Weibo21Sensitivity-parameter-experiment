"""Atomic, training-only checkpoints at a completed optimizer update.

These files carry no validation threshold and are never eligible for frozen
evaluation or model selection. The existing model/OOF contracts still apply.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

SCHEMA = "optimizer_boundary_v1"


def sampler_identity(sampler):
    """Fingerprint the deterministic index order independently of its cursor."""
    lengths = tuple(int(value) for value in sampler.lengths)
    return dict(seed=int(sampler.seed), batch_size=int(sampler.batch_size),
                megabucket_batches=int(sampler.megabucket_batches),
                drop_last=bool(sampler.drop_last), examples=len(lengths),
                lengths_sha256=hashlib.sha256(json.dumps(lengths).encode()).hexdigest())


def validate_progress(checkpoint, *, total_batches=None, accumulation=None,
                      sampler_sha256=None, runtime_sha256=None,
                      expected_sampler_identity=None):
    state = checkpoint.get("training_state", {})
    progress = state.get("progress")
    if not isinstance(progress, dict) or progress.get("schema") != SCHEMA:
        raise ValueError("not an optimizer-boundary progress checkpoint")
    if checkpoint.get("threshold_selection") is not None:
        raise ValueError("training progress must not claim a frozen threshold")
    if checkpoint.get("optimizer_state_dict") is None or checkpoint.get("scheduler_state_dict") is None:
        raise ValueError("training progress requires optimizer and scheduler states")
    epoch, batch, total, accum = (progress.get(k) for k in
                                ("epoch", "batch_in_epoch", "total_batches", "accumulation"))
    if any(type(x) is not int for x in (epoch, batch, total, accum)):
        raise ValueError("invalid progress counters")
    if epoch < 1 or accum < 1 or not 0 < batch <= total:
        raise ValueError("progress is outside its epoch")
    if batch % accum and batch != total:
        raise ValueError("progress is not at an optimizer boundary")
    if checkpoint.get("epoch") != epoch or state.get("batch_count") != batch:
        raise ValueError("progress epoch/batch statistics mismatch")
    if type(state.get("global_step")) is not int or state["global_step"] < 1:
        raise ValueError("invalid progress optimizer step")
    expected_step = (epoch - 1) * math.ceil(total / accum) + math.ceil(batch / accum)
    if state["global_step"] != expected_step:
        raise ValueError("progress optimizer step disagrees with epoch/batch")
    for actual, expected, name in (
        (total, total_batches, "total batches"),
        (accum, accumulation, "accumulation"),
        (progress.get("sampler_sha256"), sampler_sha256, "sampler implementation"),
        (progress.get("runtime_sha256"), runtime_sha256, "runtime implementation"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"progress {name} changed")
    if len(state.get("rng_states", [])) != 1:
        raise ValueError("progress resume requires one saved RNG state")
    rng = state["rng_states"][0]
    if (not isinstance(rng, dict) or not {"python", "numpy", "torch", "cuda"} <= rng.keys()
            or not torch.is_tensor(rng["torch"]) or rng["torch"].dtype != torch.uint8
            or rng["torch"].ndim != 1 or rng["torch"].numel() == 0):
        raise ValueError("invalid progress RNG state")
    identity = progress.get("sampler_identity")
    if not isinstance(identity, dict):
        raise ValueError("progress requires a deterministic sampler identity")
    if expected_sampler_identity is not None and identity != expected_sampler_identity:
        raise ValueError("progress sample order changed")
    examples, batch_size = identity.get("examples"), identity.get("batch_size")
    if (type(examples) is not int or examples < 1 or type(batch_size) is not int
            or batch_size < 1):
        raise ValueError("invalid progress sampler dimensions")
    expected_batches = (examples // batch_size if identity.get("drop_last")
                        else math.ceil(examples / batch_size))
    if total != expected_batches or state.get("epoch_examples") != min(batch * batch_size, examples):
        raise ValueError("progress sample/batch counts disagree")
    running = state.get("running")
    if (not isinstance(running, dict) or "total" not in running
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in running.values())):
        raise ValueError("invalid progress running statistics")
    elapsed = state.get("epoch_train_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid progress elapsed time")
    return progress


def select_resume_checkpoint(directory):
    """Prefer the most advanced valid checkpoint; never silently discard corruption."""
    directory = Path(directory)
    choices = []
    for name in ("last.pth", "progress.pth"):
        path = directory / name
        if not path.is_file():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if name == "progress.pth":
            validate_progress(saved)
        else:
            from mmfnd.evaluation import checkpoint_threshold_selection
            checkpoint_threshold_selection(saved)
            if saved.get("optimizer_state_dict") is None or saved.get("scheduler_state_dict") is None:
                raise ValueError("last checkpoint is not resumable")
        step = int(saved.get("training_state", {}).get("global_step", 0))
        # At an equal optimizer step, prefer last.pth with completed validation.
        choices.append((step, name == "last.pth", path))
        del saved
    return max(choices)[-1] if choices else None


def save_progress(path, model, optimizer, scheduler, epoch, config, scaler, state):
    from mmfnd.engine import save_checkpoint
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    started = time.monotonic()
    try:
        save_checkpoint(temporary, model, optimizer, scheduler, epoch, {}, config,
                        scaler=scaler, training_state=state)
        # Verify before replacing a previous valid checkpoint. Failed writes or
        # malformed boundary state cannot destroy the latest resumable file.
        validate_progress(torch.load(temporary, map_location="cpu", weights_only=False, mmap=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return time.monotonic() - started


def deadline_reached():
    value = os.environ.get("LEDGER_TRAIN_STOP_AT")
    if not value:
        return False
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        raise ValueError("LEDGER_TRAIN_STOP_AT requires a timezone")
    return datetime.now(deadline.tzinfo) >= deadline
