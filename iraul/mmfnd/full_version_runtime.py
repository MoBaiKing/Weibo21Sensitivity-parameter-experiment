"""Scoped missing-image execution for the complete Weibo21 records.

The masked R1 model, losses and reference subset game remain their unchanged
oracles. Missing images are absent observations: only T is available, and no
placeholder image enters the processor or SigLIP. All-present batches call the
original collator and encoder. This module is pinned separately by the queue.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

VERSION = "full_version_availability_mask_v1"
POLICY = "availability_mask_v1"
_DEPTH = 0
_PERMANENT_SCOPES = []


def enabled(config):
    return config.get("data", {}).get("missing_image_policy") == POLICY


def record_availability(record):
    images = record.get("images")
    if not isinstance(images, list):
        raise ValueError("full_version images must be a list, possibly empty")
    has_image = bool(images)
    return [True, has_image, has_image, has_image]


def input_availability(batch):
    count = len(batch["text_input_ids"])
    owners = batch["image_owner"]
    pixels = batch["pixel_values"]
    if owners.ndim != 1 or len(owners) != len(pixels):
        raise ValueError("image owners must match actual image tensors")
    if owners.dtype != torch.long or (owners < 0).any() or (owners >= count).any():
        raise ValueError("invalid image ownership")
    has_image = torch.zeros(count, dtype=torch.bool, device=owners.device)
    has_image[owners] = True
    return torch.stack([torch.ones_like(has_image), has_image, has_image, has_image], -1)


def _text_only_encoder(encoder, batch, ablate_component):
    # This is exactly the oracle's text-encoder/projection/readout path.
    text_out = encoder.text_encoder(
        input_ids=batch["text_input_ids"],
        attention_mask=batch["text_attention_mask"], return_dict=True,
    )
    tokens = encoder.text_proj(text_out.last_hidden_state)
    if getattr(encoder, "general_last_valid_index", False):
        mask = batch["text_attention_mask"].bool()
        if not mask.any(-1).all():
            raise ValueError("empty token sequence")
        positions = torch.arange(mask.shape[1], device=mask.device)[None]
        last = positions.expand_as(mask).masked_fill(~mask, -1).amax(-1)
    else:
        last = (batch["text_attention_mask"].sum(1) - 1).clamp_min(0)
    text = tokens[torch.arange(len(tokens), device=tokens.device), last]
    if ablate_component == "text":
        tokens, text = torch.zeros_like(tokens), torch.zeros_like(text)
    zero = torch.zeros_like(text)
    scalar = text.new_zeros(len(text))
    intrinsic = {key: scalar for key in (
        "text_relation_ambiguity", "event_contradiction", "visual_evidence_inconsistency",
        "visual_source_inconsistency", "multi_image_dispersion",
    )}
    intrinsic.update(feature=zero, multi_image_available=torch.zeros_like(scalar, dtype=torch.bool),
                     alignment_attention=text.new_empty((len(text), tokens.shape[1], 0)))
    return dict(text=text, text_tokens=tokens, vision=zero,
                vision_tokens=text.new_empty((0, 0, text.shape[-1])),
                per_image=text.new_empty((0, text.shape[-1])), intrinsic=intrinsic,
                intrinsic_feature=zero, interaction=zero,
                event_matched_similarity=scalar, event_mismatched_similarity=scalar,
                source_matched_similarity=text.new_empty((0,)),
                source_mismatched_similarity=text.new_empty((0,)),
                source_ranking_valid=torch.empty((0,), dtype=torch.bool, device=text.device))


@torch.no_grad()
def predict_student(model, loader, device, precision="fp32", deletions=False):
    """The unchanged prediction/threshold protocol with explicit real views."""
    from mmfnd.engine import autocast_context
    from mmfnd.evaluation_masked_r1 import deduplicate_rows

    model.eval()
    rows, started = [], time.monotonic()
    for raw in loader:
        inputs = {key: raw[key].to(device, non_blocking=True) for key in
                  ("text_input_ids", "text_attention_mask", "pixel_values", "image_owner")}
        available = input_availability(inputs)
        supplied = raw.get("availability")
        if supplied is None or not torch.equal(supplied.to(available.device).bool(), available):
            raise ValueError("prediction availability differs from observed images")
        with autocast_context(device, precision):
            z = model.encode_views(inputs)
            out = model.predict_views(z, None if available.all() else available)
            deleted = []
            if deletions:
                for index in range(4):
                    retained = available.clone()
                    retained[:, index] = False
                    legal = retained.any(-1)
                    values = [None] * len(z)
                    if legal.any():
                        probabilities = model.predict_views(z[legal], retained[legal])["logits"].float().softmax(-1).cpu().tolist()
                        for row_index, value in zip(legal.nonzero().flatten().tolist(), probabilities):
                            values[row_index] = value
                    deleted.append(values)
        cpu = dict(probabilities=out["logits"].float().softmax(-1).cpu().tolist(),
                   utility=out["utility"].float().cpu().tolist(),
                   weights=out["final_evidence_weights"].float().cpu().tolist(),
                   relation_probs=out["relation_probs"].float().cpu().tolist(),
                   pair_available=out["pair_available"].cpu().tolist())
        if "relation_vacuity" in out:
            cpu["relation_vacuity"] = out["relation_vacuity"].cpu().tolist()
        for index, sid in enumerate(raw["ids"]):
            row = dict(id=str(sid), **{key: value[index] for key, value in cpu.items()})
            if deletions:
                row["deleted_probabilities"] = [values[index] for values in deleted]
            rows.append(row)
    return deduplicate_rows(rows), dict(inference_seconds=time.monotonic() - started)


@contextmanager
def installed(config):
    """Enable only the explicit full_version policy; nested scopes are safe."""
    global _DEPTH
    report = dict(enabled=enabled(config), version=VERSION, policy=POLICY,
                  module="mmfnd/full_version_runtime.py",
                  module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not report["enabled"]:
        yield report
        return
    if config.get("dataset", {}).get("name") != "weibo21":
        raise ValueError("full_version runtime is restricted to the Weibo21 dataset")
    if config.get("model", {}).get("architecture_version") != "qwen_lora_lgled_masked_r1":
        raise ValueError("full_version runtime requires the unchanged masked R1 architecture")
    if _DEPTH:
        _DEPTH += 1
        try:
            yield report
        finally:
            _DEPTH -= 1
        return

    from mmfnd import data, dataset_contract, evaluation_masked_r1, evaluation_runtime, model, r1_data
    base_init = data.CUTEFNDMultimodalDataset.__init__
    base_collate = data.CUTEFNDMultimodalDataset.collate_fn
    replay_collate = r1_data.ReplayDataset.collate_fn
    encoder_forward = model.MultimodalIntrinsicEvidenceEncoder.forward

    def dataset_init(dataset, *args, **kwargs):
        base_init(dataset, *args, **kwargs)
        data_config = args[3] if len(args) > 3 else kwargs["config"]
        dataset._full_version_missing_images = data_config.get("missing_image_policy") == POLICY

    def collate(dataset, batch):
        if not getattr(dataset, "_full_version_missing_images", False):
            return base_collate(dataset, batch)
        if not batch:
            raise ValueError("empty full_version batch")
        if any(item["images"] for item in batch):
            out = base_collate(dataset, batch)
        else:
            texts = [f"{dataset.text_instruction}{item['text']}" for item in batch]
            tokens = dataset.processor.tokenizer(texts, padding=True, truncation=True,
                                                 max_length=dataset.max_text_length, return_tensors="pt")
            out = dict(ids=[item["id"] for item in batch], text_input_ids=tokens["input_ids"],
                       text_attention_mask=tokens["attention_mask"],
                       pixel_values=torch.empty((0, 3, 0, 0), dtype=torch.float32),
                       image_owner=torch.empty((0,), dtype=torch.long),
                       labels=torch.tensor([item["label"] for item in batch], dtype=torch.long),
                       image_paths=[item["image_paths"] for item in batch],
                       texts=[item["text"] for item in batch], categories=[item["category"] for item in batch])
        out["availability"] = input_availability(out)
        return out

    def replay(dataset, batch):
        if not getattr(dataset, "_full_version_missing_images", False):
            return replay_collate(dataset, batch)
        out = collate(dataset, batch)
        out["augmentation_ids"] = [item["augmentation_id"] for item in batch]
        return out

    def require_multi_image_records(manifest_dir, dataset_name):
        multiple = False
        for split in ("train", "val", "test"):
            path = Path(manifest_dir) / f"{split}.jsonl"
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("images"), list):
                    raise ValueError(f"{path}:{number} images must be a list, possibly empty")
                multiple |= len(row["images"]) > 1
        if not multiple:
            raise ValueError(f"{dataset_name} must retain every real image, including multiple-image records")

    def encoder(encoder, batch, ablate_component=None):
        available = input_availability(batch)
        if not len(batch["image_owner"]):
            return _text_only_encoder(encoder, batch, ablate_component)
        encoded = encoder_forward(encoder, batch, ablate_component)
        if available.all():
            return encoded
        missing = ~available[:, 1]
        encoded = dict(encoded)
        for key in ("vision", "intrinsic_feature", "interaction"):
            encoded[key] = encoded[key].masked_fill(missing[:, None], 0)
        intrinsic = dict(encoded["intrinsic"])
        for key, value in intrinsic.items():
            if torch.is_tensor(value) and value.ndim and len(value) == len(missing):
                mask = missing.reshape((len(missing),) + (1,) * (value.ndim - 1))
                intrinsic[key] = value.masked_fill(mask, 0)
        encoded["intrinsic"] = intrinsic
        return encoded

    with ExitStack() as stack:
        for owner, name, value in (
            (data.CUTEFNDMultimodalDataset, "__init__", dataset_init),
            (data.CUTEFNDMultimodalDataset, "collate_fn", collate),
            (r1_data.ReplayDataset, "collate_fn", replay),
            (dataset_contract, "_require_multi_image_records", require_multi_image_records),
            (model.MultimodalIntrinsicEvidenceEncoder, "forward", encoder),
            (evaluation_masked_r1, "predict_student", predict_student),
            (evaluation_runtime, "predict_student", predict_student),
        ):
            stack.enter_context(patch.object(owner, name, value))
        _DEPTH = 1
        try:
            yield report
        finally:
            _DEPTH = 0


def install_full_version_runtime(config):
    """Keep the scope for a single-experiment prepare process until it exits."""
    scope = installed(config)
    report = scope.__enter__()
    _PERMANENT_SCOPES.append(scope)
    return report
