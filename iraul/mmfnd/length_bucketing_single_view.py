"""Execution-only single-view coverage on top of the bucketed sampler.

The formal R1 protocol materialises every training record twice per epoch
(``clean`` and ``aug1``; see ``mmfnd/r1_data.py``), which doubles the number of
micro-batches per epoch.  This module lets an explicitly configured run train on
exactly one view per record per epoch while keeping the *same* reference target
cache: the OOF cache stores one row per ``(sample_id, augmentation_id)`` pair,
so a single-view epoch simply consumes one of the two rows that already exist
and never changes which supervision a view carries.

Design constraints (do not relax them without re-deriving the fingerprints):

* It is deliberately a separate module.  ``mmfnd/length_bucketing.py`` is hashed
  into ``config["length_bucketing_effective"]["module_sha256"]`` and that hash is
  compared against every existing ``progress.pth`` on resume, so editing it
  would break mid-epoch resume for runs that are already in flight.
* It must never be referenced from ``model.*`` or ``data.*``: those are hashed
  into the reference OOF identity.  ``train.length_bucketing_view_policy`` is
  not part of ``experiment_fingerprint`` and therefore leaves every
  ``targets.pt`` valid.
* View selection uses no RNG at all, so it cannot perturb the replay pool,
  model dropout or data-loader streams.
* The default (policy absent) path is byte-for-byte the existing sampler.

Protocol names
--------------
``single_view_per_epoch_v1`` with ``mode``:

* ``alternate_epoch`` (default) — record *i* uses variant ``(i + epoch) % 2``,
  i.e. exactly half of every epoch is ``clean`` and half is ``aug1``, and each
  record alternates views across consecutive epochs.
* ``hash_balanced`` — variant chosen from ``sha256(seed:epoch:record)`` bit 0.
* ``clean`` / ``augmented`` — every record uses that single view (diagnostics).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

from mmfnd.length_bucketing import LengthBucketBatchSampler

SCHEMA = "single_view_per_epoch_v1"
VIEWS_PER_EPOCH = 1
CLEAN_VARIANT = 0
AUGMENTED_VARIANT = 1
MODES = ("alternate_epoch", "hash_balanced", "clean", "augmented")


@dataclass(frozen=True)
class ViewPolicy:
    """Validated view-selection policy for one configured run."""

    schema: str
    mode: str
    record_count: int

    @property
    def identity(self) -> dict:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "views_per_epoch": VIEWS_PER_EPOCH,
            "records_per_epoch": self.record_count,
            "samples_per_epoch": self.record_count,
            "uses_rng": False,
        }


def parse_view_policy(raw: object, lengths: Sequence[int]) -> ViewPolicy:
    """Validate one ``train.length_bucketing_view_policy`` mapping."""
    if not isinstance(raw, dict):
        raise ValueError("length_bucketing_view_policy must be a mapping")
    schema = str(raw.get("schema", SCHEMA))
    if schema != SCHEMA:
        raise ValueError(f"unsupported view policy schema: {schema}")
    mode = str(raw.get("mode", "alternate_epoch"))
    if mode not in MODES:
        raise ValueError(f"unsupported view policy mode: {mode}")
    if len(lengths) % 2:
        raise ValueError(
            "single-view coverage requires the clean+aug1 replay pool "
            f"(even sample count), got {len(lengths)}"
        )
    record_count = len(lengths) // 2
    if record_count < 1:
        raise ValueError("view policy requires at least one training record")
    return ViewPolicy(schema=schema, mode=mode, record_count=record_count)


def variant_for(policy: ViewPolicy, record: int, epoch: int, seed: int) -> int:
    """Deterministic variant index (0 = clean, 1 = aug1) for one record."""
    if policy.mode == "clean":
        return CLEAN_VARIANT
    if policy.mode == "augmented":
        return AUGMENTED_VARIANT
    if policy.mode == "alternate_epoch":
        return (int(record) + int(epoch)) % 2
    digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{int(record)}".encode()).digest()
    return digest[0] & 1


def selected_dataset_indices(policy: ViewPolicy, lengths: Sequence[int],
                             epoch: int, seed: int) -> list[int]:
    """Dataset indices for one single-view epoch, in record order."""
    return [
        2 * record + variant_for(policy, record, epoch, seed)
        for record in range(policy.record_count)
    ]


class SingleViewBatchSampler(LengthBucketBatchSampler):
    """One view per record per epoch, reusing the bucketed batch layout.

    ``lengths`` is the full clean+aug1 pool (dataset index order), so the
    parent's algorithm can be reused verbatim on the selected subset.  The
    instance exposes the *selected* lengths through ``self.lengths`` because
    ``mmfnd.training_progress.sampler_identity`` reads that attribute and the
    identity must describe the epoch that actually runs.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, *,
                 megabucket_batches: int = 100, drop_last: bool = False,
                 view_policy: object | None = None) -> None:
        pool = [int(value) for value in lengths]
        self.policy = parse_view_policy(view_policy, pool)
        self.pool_lengths = tuple(pool)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.megabucket_batches = int(megabucket_batches)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._refresh()

    def _refresh(self) -> None:
        self.dataset_indices = selected_dataset_indices(
            self.policy, self.pool_lengths, self.epoch, self.seed
        )
        # The delegated sampler owns the bucketing/shuffle; positions are
        # remapped back to dataset indices by __iter__.
        self.lengths = tuple(self.pool_lengths[index] for index in self.dataset_indices)
        self._delegate = LengthBucketBatchSampler(
            self.lengths, self.batch_size, self.seed,
            megabucket_batches=self.megabucket_batches, drop_last=self.drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._refresh()

    @property
    def start_batch(self) -> int:
        return self._delegate.start_batch

    @start_batch.setter
    def start_batch(self, value: int) -> None:
        self._delegate.start_batch = value

    def __len__(self) -> int:
        return len(self._delegate)

    def __iter__(self) -> Iterator[list[int]]:
        indices = self.dataset_indices
        for batch in self._delegate:
            yield [indices[position] for position in batch]

    def identity(self) -> dict:
        """Auditable description recorded in the run config and progress state."""
        return {
            **self.policy.identity,
            "module": "mmfnd/length_bucketing_single_view.py",
            "dataset_indices_sha256": hashlib.sha256(
                ",".join(map(str, self.dataset_indices)).encode()
            ).hexdigest(),
            "pool_examples": len(self.pool_lengths),
            "examples_this_epoch": len(self.lengths),
        }


def build_single_view_bucketed_loader(source, lengths: Sequence[int],
                                      batch_size: int, seed: int, *,
                                      megabucket_batches: int,
                                      view_policy: object):
    """Clone ``source`` with the single-view bucketed sampler.

    ``torch.utils.data.DataLoader`` refuses to replace ``batch_sampler`` after
    construction, so the loader is rebuilt exactly the way
    ``mmfnd.length_bucketing.build_length_bucketed_loader`` rebuilds it (same
    worker, pinning, generator and context options) -- only the sampler differs.
    ``mmfnd/length_bucketing.py`` itself stays untouched because its hash is
    recorded in every existing ``progress.pth``.
    """
    from torch.utils.data import DataLoader

    sampler = SingleViewBatchSampler(
        lengths, batch_size, seed,
        megabucket_batches=megabucket_batches, drop_last=bool(source.drop_last),
        view_policy=view_policy,
    )
    if len(sampler.lengths) != sampler.policy.record_count:
        raise RuntimeError("single-view sampler produced an unexpected epoch size")
    options = {
        "dataset": source.dataset,
        "batch_sampler": sampler,
        "num_workers": source.num_workers,
        "collate_fn": source.collate_fn,
        "pin_memory": source.pin_memory,
        "timeout": source.timeout,
        "worker_init_fn": source.worker_init_fn,
        "multiprocessing_context": source.multiprocessing_context,
        "generator": source.generator,
    }
    if source.num_workers > 0:
        options.update(
            persistent_workers=source.persistent_workers,
            prefetch_factor=source.prefetch_factor,
        )
    pin_memory_device = getattr(source, "pin_memory_device", "")
    if pin_memory_device:
        options["pin_memory_device"] = pin_memory_device
    return DataLoader(**options), sampler.identity()

