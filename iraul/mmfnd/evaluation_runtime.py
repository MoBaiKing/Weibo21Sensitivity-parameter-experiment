"""Batch host transfers for the unchanged R1 prediction and threshold protocol."""
from pathlib import Path
import hashlib
import time

import torch


@torch.no_grad()
def predict_student(model, loader, device, precision="fp32", deletions=False):
    from mmfnd.engine import autocast_context
    from mmfnd.evaluation_masked_r1 import deduplicate_rows
    model.eval()
    rows = []
    started = time.monotonic()
    for raw in loader:
        inputs = {key: raw[key].to(device, non_blocking=True) for key in
                  ("text_input_ids", "text_attention_mask", "pixel_values", "image_owner")}
        with autocast_context(device, precision):
            z = model.encode_views(inputs)
            out = model.predict_views(z)
            deleted = []
            if deletions:
                for index in range(4):
                    available = torch.ones(z.shape[:2], device=device, dtype=torch.bool)
                    available[:, index] = False
                    deleted.append(model.predict_views(z, available)["logits"].float().softmax(-1).cpu())
        # Copy once per field rather than once per sample and field. Keep the
        # oracle dtypes and synchronous D2H completion before reading Python rows.
        cpu = dict(
            probabilities=out["logits"].float().softmax(-1).cpu().tolist(),
            utility=out["utility"].float().cpu().tolist(),
            weights=out["final_evidence_weights"].float().cpu().tolist(),
            relation_probs=out["relation_probs"].float().cpu().tolist(),
            pair_available=out["pair_available"].cpu().tolist(),
        )
        if "relation_vacuity" in out:
            cpu["relation_vacuity"] = out["relation_vacuity"].cpu().tolist()
        deleted = [value.tolist() for value in deleted]
        for index, sid in enumerate(raw["ids"]):
            row = dict(id=str(sid), **{key: value[index] for key, value in cpu.items()})
            if deletions:
                row["deleted_probabilities"] = [value[index] for value in deleted]
            rows.append(row)
    return deduplicate_rows(rows), dict(inference_seconds=time.monotonic() - started)


def install_prediction_runtime():
    from mmfnd import evaluation_masked_r1
    evaluation_masked_r1.predict_student = predict_student
    return dict(version="batched_prediction_transfer_v1", module="mmfnd/evaluation_runtime.py",
                module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
