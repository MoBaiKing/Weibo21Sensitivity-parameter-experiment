"""Linear-time strict OOF cache validation, retaining the original lookup API.

The fingerprinted implementation remains the protocol oracle. Fold manifests
and checkpoint bytes are verified once per referenced fold, and probabilities
are checked in bounded tensor batches instead of thousands of tiny CPU kernels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mmfnd.r1_cache import TargetCache as OriginalTargetCache
from mmfnd.r1_cache import file_sha, validate_fold, validate_probabilities

VERSION = "strict_cache_linear_validation_v1"


class TargetCache(OriginalTargetCache):
    """The same strict provenance checks and inherited exact augmentation join."""

    def __init__(self, path, expected, *, probability_chunk_size=4096):
        if not isinstance(probability_chunk_size, int) or probability_chunk_size < 1:
            raise ValueError("probability validation chunk size must be positive")
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(path.with_suffix(".json").read_text())
        if manifest["payload_sha256"] != file_sha(path):
            raise ValueError("target cache payload hash mismatch")
        if payload["fingerprint"] != expected:
            raise ValueError("cache preprocess/model/split/reference seed mismatch")
        self.scale = float(payload["scale"])
        self.rows = payload["rows"]
        self.epsilon = float(payload.get("epsilon", 1e-7))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("invalid training-only scale")
        self.fingerprint = expected
        self.index = {}
        fold_contracts = {}
        for row in self.rows:
            key = (str(row["sample_id"]), row["augmentation_id"])
            if key in self.index:
                raise ValueError("duplicate cached augmentation")
            number = row["fold"]
            if number not in fold_contracts:
                fold = expected["folds"]["folds"][number]
                validate_fold(fold)
                checkpoint_path = path.parent / f"fold_{number}.pth"
                if not checkpoint_path.is_file():
                    raise ValueError("missing reference checkpoint provenance")
                fold_contracts[number] = (
                    set(fold["target_ids"]), fold["fit_id_hash"], fold["cal_id_hash"], file_sha(checkpoint_path),
                )
            target_ids, fit_hash, cal_hash, checkpoint_hash = fold_contracts[number]
            if key[0] not in target_ids:
                raise ValueError("not a held-out prediction")
            if row["fit_id_hash"] != fit_hash or row["cal_id_hash"] != cal_hash:
                raise ValueError("reference label-use hashes mismatch")
            if row["reference_checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("reference checkpoint differs from cached OOF predictions")
            self.index[key] = row
        for start in range(0, len(self.rows), probability_chunk_size):
            rows = self.rows[start:start + probability_chunk_size]
            # Stacking changes no arithmetic in validate_probabilities: bounds,
            # finiteness and the class-axis sum retain the original tolerances.
            # Group by dtype/shape to avoid implicit promotion changing the
            # original per-row sum tolerance for mixed-dtype legacy artifacts.
            groups = {}
            for row in rows:
                probability = torch.as_tensor(row["q"])
                groups.setdefault((probability.dtype, tuple(probability.shape)), []).append(probability)
            for probabilities in groups.values():
                validate_probabilities(torch.stack(probabilities))
