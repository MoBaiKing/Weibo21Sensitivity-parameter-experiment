"""Deterministic length-bucketed batches for single-process training.

This module only changes the order in which samples are presented.  It keeps
the dataset, collate function, worker settings, and batch size of the source
DataLoader intact while grouping similarly sized token sequences to reduce
padding.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import DataLoader, Sampler


def sample_token_lengths(
    dataset,
    tokenizer,
    text_instruction: str,
    max_text_length: int,
    *,
    tokenizer_batch_size: int = 2048,
) -> list[int]:
    """Return the exact post-truncation token length for every dataset row."""
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("length bucketing requires dataset.records")
    if tokenizer_batch_size <= 0:
        raise ValueError("tokenizer_batch_size must be positive")

    lengths: list[int] = []
    for start in range(0, len(records), tokenizer_batch_size):
        texts = [
            f"{text_instruction}{record['text']}"
            for record in records[start : start + tokenizer_batch_size]
        ]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=int(max_text_length),
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)

    # ReplayDataset exposes clean/augmented views as adjacent indices while
    # keeping one source record.  Both variants have the same text length.
    if lengths and len(dataset) != len(lengths):
        if len(dataset) % len(lengths):
            raise RuntimeError(
                f"token length count mismatch: {len(lengths)} source rows for "
                f"{len(dataset)} dataset items"
            )
        repeat = len(dataset) // len(lengths)
        lengths = [value for value in lengths for _ in range(repeat)]
    if len(lengths) != len(dataset):
        raise RuntimeError(
            f"token length count mismatch: {len(lengths)} != {len(dataset)}"
        )
    if any(value <= 0 or value > int(max_text_length) for value in lengths):
        raise ValueError("token lengths must be within the configured truncation limit")
    return lengths


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle full batches while keeping the sole incomplete batch last.

    The trainer weights gradient accumulation windows by their actual sample
    count and assumes that only the final micro-batch can be incomplete.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int,
        *,
        megabucket_batches: int = 100,
        drop_last: bool = False,
    ) -> None:
        if not lengths:
            raise ValueError("lengths must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if megabucket_batches <= 0:
            raise ValueError("megabucket_batches must be positive")
        self.lengths = tuple(int(value) for value in lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.megabucket_batches = int(megabucket_batches)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        count = (len(self.lengths) // self.batch_size if self.drop_last
                 else math.ceil(len(self.lengths) / self.batch_size))
        if not isinstance(self.start_batch, int) or not 0 <= self.start_batch <= count:
            raise ValueError("start_batch must be an integer within the epoch")
        return count - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        len(self)  # Validate the resume cursor before yielding any samples.
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        mega_size = self.batch_size * self.megabucket_batches
        batches: list[list[int]] = []
        tail: list[int] | None = None
        for offset in range(0, len(indices), mega_size):
            megabucket = indices[offset : offset + mega_size]
            megabucket.sort(key=self.lengths.__getitem__, reverse=True)
            for batch_offset in range(0, len(megabucket), self.batch_size):
                batch = megabucket[batch_offset : batch_offset + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
                elif not self.drop_last:
                    tail = batch
        rng.shuffle(batches)
        if tail is not None:
            batches.append(tail)
        # Recreate the same full epoch before slicing. A resumed loader need
        # not load images or tokenize any samples from completed updates.
        yield from batches[self.start_batch:]


def build_length_bucketed_loader(
    source: DataLoader,
    lengths: Sequence[int],
    batch_size: int,
    seed: int,
    *,
    megabucket_batches: int = 100,
) -> DataLoader:
    """Clone a DataLoader with a deterministic length-bucket batch sampler."""
    if len(lengths) != len(source.dataset):
        raise ValueError("one token length is required for each dataset row")
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size,
        seed,
        megabucket_batches=megabucket_batches,
        drop_last=bool(source.drop_last),
    )
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
    return DataLoader(**options)
