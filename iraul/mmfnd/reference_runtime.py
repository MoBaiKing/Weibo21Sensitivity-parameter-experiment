"""Execution-only reference acceleration; the fingerprinted pipeline is the oracle.

No fold, sample order, augmentation, loss, or checkpoint payload is changed.
The entry-point wrapper records this module separately in runtime metadata.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from mmfnd.engine import autocast_context
from mmfnd.reference_subset_probe import subset_probabilities

VERSION = "reference_io_runtime_v1"
INPUT_KEYS = ("text_input_ids", "text_attention_mask", "pixel_values", "image_owner")


def move_batch(batch, device):
    """Queue pinned host copies on the current stream without per-tensor waits."""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def input_tensors(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in INPUT_KEYS}


def apply_model_runtime(model, config):
    """Honor the same explicit execution override used by student training."""
    requested = config.get("train", {}).get("qwen_gradient_checkpointing_override")
    if requested is None:
        return None
    base = model.encoder.shared_qwen_model()
    enabled = bool(getattr(base, "gradient_checkpointing", False))
    if bool(requested) and not enabled:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif not bool(requested) and enabled:
        base.gradient_checkpointing_disable()
    return bool(getattr(base, "gradient_checkpointing", False))


def partition_loader(dataset, ids, config, context, train=False):
    """Keep the oracle sampler and RNG draws, adding only pinned I/O/prefetch."""
    selected = set(ids)
    factor = 2 if dataset.pool else 1
    indices = [i for i in range(len(dataset)) if str(dataset.records[i // factor]["id"]) in selected]
    subset = Subset(dataset, indices)
    sampler = (DistributedSampler(subset, num_replicas=context.world_size, rank=context.rank,
                                  shuffle=True) if train and context.distributed else None)
    workers = int(config["data"]["num_workers"])
    options = {}
    if workers:
        options["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    # Persistent workers change iterator base-seed draws and thus the following
    # epoch's default RandomSampler order. Preserve the original lifecycle.
    return DataLoader(subset, batch_size=config["train"]["per_gpu_batch_size"],
                      shuffle=train and sampler is None, sampler=sampler,
                      num_workers=workers, collate_fn=dataset.collate_fn,
                      pin_memory=bool(config["data"].get("pin_memory", False)),
                      persistent_workers=False, **options)


@torch.no_grad()
def predict_reference(model, loader, device, precision, prior, temperature=1.):
    """Transfer the complete B x 16 x 2 probability tensor once per batch."""
    model.eval()
    rows = []
    for raw in loader:
        if "labels" in raw or "label" in raw:
            raise ValueError("predict_reference requires label-free input batches")
        with autocast_context(device, precision):
            z = model.encode(input_tensors(raw, device))
            q = subset_probabilities(model.probe, z, prior, temperature)
        # One blocking D2H copy makes every subsequent CPU row immediately safe.
        # Do not use asynchronous D2H without a completion event here.
        q = q.cpu()
        available = raw["availability"].tolist()
        for i, (sid, aid) in enumerate(zip(raw["ids"], raw["augmentation_ids"])):
            rows.append(dict(sample_id=str(sid), augmentation_id=aid,
                             availability=available[i], q=q[i]))
    return rows


@torch.no_grad()
def calibration_logits(model, loader, device, precision):
    """Keep labels and availability on CPU; only model inputs cross to CUDA."""
    model.eval()
    logits, labels, availability = [], [], []
    for raw in loader:
        with autocast_context(device, precision):
            value = model(input_tensors(raw, device))
        logits.append(value.float().cpu())
        labels.append(raw["labels"].cpu())
        availability.append(raw["availability"].cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(availability)


@contextmanager
def runtime_overrides(config):
    """Scope replacements to one synchronous reference run and restore on exit.

    The original source files remain byte-for-byte intact, so existing strict
    cache identities and pretrained/fold checks are still checked normally.
    """
    from mmfnd import reference_pipeline as pipeline
    from mmfnd.r1_cache import file_sha

    names = ("ReferenceModel", "partition_loader", "predict_reference", "calibration_logits", "move_batch")
    originals = {name: getattr(pipeline, name) for name in names}
    report = dict(version=VERSION, source_sha256=file_sha(__file__),
                  pin_memory=bool(config["data"].get("pin_memory", False)),
                  prefetch_factor=int(config["data"].get("prefetch_factor", 2)),
                  persistent_partition_workers=False, sample_order_unchanged=True,
                  qwen_gradient_checkpointing_override=config.get("train", {}).get("qwen_gradient_checkpointing_override"),
                  timings={})

    def timed(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            started = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                stats = report["timings"].setdefault(name, dict(calls=0, seconds=0.))
                stats["calls"] += 1
                stats["seconds"] += time.monotonic() - started
        return call

    class RuntimeReferenceModel(originals["ReferenceModel"]):
        def __init__(self, model_config):
            super().__init__(model_config)
            report["effective_qwen_gradient_checkpointing"] = apply_model_runtime(self, config)

    try:
        pipeline.ReferenceModel = RuntimeReferenceModel
        pipeline.partition_loader = partition_loader
        pipeline.predict_reference = timed("predict_reference", predict_reference)
        pipeline.calibration_logits = timed("calibration_logits", calibration_logits)
        pipeline.move_batch = move_batch
        yield report
    finally:
        for name, original in originals.items():
            setattr(pipeline, name, original)


def build_reference(root, config, context, directory):
    """Run the unchanged fit/control flow with scoped, measured I/O replacements."""
    from mmfnd import reference_pipeline as pipeline
    from mmfnd.utils import dump_json

    started = time.monotonic()
    # Do not write into an existing reference if the oracle rejects it before
    # fitting. Existing provenance must not be modified by a failed invocation.
    manifest_existed = (Path(directory) / "reference_manifest.json").exists()
    with runtime_overrides(config) as report:
        try:
            return pipeline.build_reference(root, config, context, directory)
        finally:
            report["total_seconds"] = time.monotonic() - started
            if context.is_main and not manifest_existed and Path(directory).is_dir():
                dump_json(report, Path(directory) / "reference_runtime.json")
