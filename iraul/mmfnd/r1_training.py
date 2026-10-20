"""Execution-only acceleration for the unchanged masked R1 training contract.

The original loss remains the compatibility oracle. This module reuses its
offline targets and EDL implementation, and is fingerprinted in run metadata.
No cache validation, model contract or checkpoint source check is bypassed.
"""
from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from mmfnd.evidence_utility import contribution_targets
from mmfnd.relation_supervision import relation_targets, soft_edl_loss

VERSION = "r1_prepared_supervision_v1"


class PreparedSupervision:
    """CPU table, built only AFTER TargetCache has verified OOF provenance.

Validate stored targets against train labels and the original equations once.
The hot path performs an exact ID/augmentation/label/availability join. Online
view dropout must use the original loss, which computes the changed subgame.
"""

    def __init__(self, cache, train_records, loss_config, chunk_size=4096):
        started = time.monotonic()
        self.target_name = "delta" if loss_config.get("utility_target", "shapley") == "signed_loo" else "phi"
        epsilon = float(loss_config.get("epsilon", 1e-7))
        if epsilon != cache.epsilon or not math.isfinite(cache.scale) or cache.scale <= 0:
            raise ValueError("prepared supervision scale/epsilon mismatch")
        labels = {str(r["id"]): int(r["label"]) for r in train_records}
        if len(labels) != len(train_records) or set(labels.values()) - {0, 1}:
            raise ValueError("duplicate training IDs or invalid labels")
        rows = cache.rows
        if not rows or {str(r["sample_id"]) for r in rows} != set(labels):
            raise ValueError("prepared supervision training coverage mismatch")
        self.index = {(str(r["sample_id"]), r["augmentation_id"]): i for i, r in enumerate(rows)}
        if len(self.index) != len(rows):
            raise ValueError("duplicate cached augmentation")
        self.labels = torch.tensor([labels[str(r["sample_id"])] for r in rows], dtype=torch.long)
        self.availability = torch.tensor([r["availability"] for r in rows], dtype=torch.bool)
        if self.availability.shape != (len(rows), 4) or not self.availability.any(-1).all():
            raise ValueError("invalid cached availability")
        relations, utilities = [], []
        for start in range(0, len(rows), chunk_size):
            part = rows[start:start + chunk_size]
            q = torch.stack([torch.as_tensor(r["q"]) for r in part]).cpu()
            y = self.labels[start:start + len(part)]
            a = self.availability[start:start + len(part)]
            effects = contribution_targets(q, y, a, epsilon)
            expected_relation = relation_targets(q[:, [1, 2, 4, 8], 0])
            stored_relation = torch.stack([torch.as_tensor(r["relation_target"]) for r in part]).cpu()
            stored_utility = torch.stack([torch.as_tensor(r[self.target_name]) for r in part]).cpu()
            # Cache targets were constructed with these exact CPU equations.
            # Rechecking catches stale labels and target corruption independently
            # of the payload hash already verified by TargetCache.
            torch.testing.assert_close(stored_relation, expected_relation, rtol=0, atol=0)
            torch.testing.assert_close(stored_utility, effects[self.target_name], rtol=0, atol=0)
            relations.append(stored_relation.float())
            utilities.append((stored_utility.double() / cache.scale).float())
        self.relations = torch.cat(relations)
        self.utilities = torch.cat(utilities)
        self.metadata = {
            "version": VERSION, "rows": len(rows), "utility_target": self.target_name,
            "preparation_seconds": time.monotonic() - started,
            "bytes": sum(t.numel() * t.element_size() for t in
                         (self.labels, self.availability, self.relations, self.utilities)),
            "targets_rechecked_against_train_labels": True,
        }

    def attach(self, batch):
        """Join on the CPU before H2D; labels/IDs never enter model inputs."""
        ids, aids = batch["ids"], batch["augmentation_ids"]
        if len(ids) != len(aids) or not ids:
            raise ValueError("invalid cache join lengths")
        indices = torch.tensor([self.index[(str(sid), aid)] for sid, aid in zip(ids, aids)])
        expected_a = self.availability[indices]
        if batch["labels"].device.type != "cpu" or batch["availability"].device.type != "cpu":
            raise ValueError("prepared target matching must occur before device transfer")
        if not torch.equal(batch["labels"], self.labels[indices]):
            raise ValueError("cached training label mismatch")
        if not torch.equal(batch["availability"], expected_a):
            raise ValueError("cached availability mismatch")
        return dict(batch, r1_relation_target=self.relations[indices],
                    r1_utility_target=self.utilities[indices], r1_target_availability=expected_a)


def prepared_r1_loss(output, labels, batch, cfg, epoch):
    """Same CE + lambda_rel EDL + lambda_utility SmoothL1 as masked_r1_loss.

    This execution path is tested for value AND gradient equivalence to the
    original loss. It returns detached tensors to avoid per-component GPU sync.
    """
    available, pairs = output["evidence_available"], output["pair_available"]
    if not torch.equal(available, batch["r1_target_availability"]):
        raise ValueError("prepared targets cannot be used for a changed availability subgame")
    rel_target = batch["r1_relation_target"].detach()
    target = batch["r1_utility_target"].detach()
    cls = F.cross_entropy(output["logits"].float(), labels)
    beta = float(cfg.get("beta_edl", .01)) * min(1., max(0., (epoch - 1) / max(1, int(cfg.get("edl_warmup_epochs", 3)))))
    if "relation_alpha" in output:
        relation = soft_edl_loss(output["relation_alpha"], rel_target, pairs, beta)
    else:
        losses = -(rel_target * output["relation_logits"].float().log_softmax(-1)).sum(-1)
        relation = (losses * pairs).sum() / pairs.sum().clamp_min(1)
    utility = (F.smooth_l1_loss(output["utility"], target.float(), reduction="none") * available).sum() / available.sum()
    total = cls + float(cfg.get("lambda_relation", .2)) * relation + float(cfg.get("lambda_utility", 1)) * utility
    return total, {k: v.detach() for k, v in dict(classification=cls, relation=relation, utility=utility).items()}


class PrefetchedBatches:
    """One-batch CUDA copy lookahead; data workers prepare the following batch.

    Does not invoke the model, draw random numbers, change the sampler or modify
    batch boundaries. record_stream protects copies while compute is in flight.
    """

    def __init__(self, loader, device, supervision=None):
        self.loader, self.device, self.supervision = loader, torch.device(device), supervision

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        source = iter(self.loader)
        if self.device.type != "cuda":
            for raw in source:
                batch = self.supervision.attach(raw) if self.supervision is not None else raw
                yield {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            return
        stream = torch.cuda.Stream(device=self.device)

        def preload():
            try:
                raw = next(source)
            except StopIteration:
                return None
            batch = self.supervision.attach(raw) if self.supervision is not None else raw
            with torch.cuda.stream(stream):
                return {k: (v if v.is_pinned() else v.pin_memory()).to(self.device, non_blocking=True)
                        if torch.is_tensor(v) else v for k, v in batch.items()}

        pending = preload()
        while pending is not None:
            compute = torch.cuda.current_stream(self.device)
            compute.wait_stream(stream)
            batch = pending
            for value in batch.values():
                if torch.is_tensor(value):
                    value.record_stream(compute)
            # Only wait for this batch, then enqueue the next copy before
            # returning control to model compute on the current stream.
            pending = preload()
            yield batch
