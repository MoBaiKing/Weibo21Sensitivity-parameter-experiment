"""Cache deterministic text encoding without changing image or replay handling."""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

VERSION = "cached_text_tokenization_v1"


class CachedTokenizer:
    """Use the original tokenizer's padding on its own cached token sequences.

    Only the exact dataset collate call is accelerated. Other tokenizer calls
    (including length inspection) retain their original implementation.
    """

    def __init__(self, tokenizer, encodings, max_length):
        self.tokenizer = tokenizer
        self.encodings = encodings
        self.max_length = max_length
        self.signature = self._signature()

    def _signature(self):
        return (self.tokenizer.padding_side, self.tokenizer.truncation_side,
                self.tokenizer.pad_token_id, tuple(self.tokenizer.model_input_names))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "tokenizer"), name)

    def __call__(self, texts, *args, **kwargs):
        exact_call = dict(padding=True, truncation=True, max_length=self.max_length,
                          return_tensors="pt")
        if (args or kwargs != exact_call or not isinstance(texts, (list, tuple))
                or not texts or self._signature() != self.signature
                or any(text not in self.encodings for text in texts)):
            return self.tokenizer(texts, *args, **kwargs)
        # Copy the small lists so padding implementations cannot modify cached
        # rows, including when the same text occurs more than once in a batch.
        rows = [{key: list(value) for key, value in self.encodings[text].items()}
                for text in texts]
        return self.tokenizer.pad(rows, padding=True, return_tensors="pt")


def enable_cached_tokenization(loader, *, chunk_size=2048):
    """Install before workers start; preserve the dataset's original collator.

    The processor is shallow-copied because train and validation datasets may
    share it. Token IDs depend on raw text and the exact unchanged tokenizer,
    not labels, augmentation, sample ordering, or the random number generators.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if getattr(loader, "_iterator", None) is not None:
        raise ValueError("install tokenization cache before DataLoader workers start")
    dataset = loader.dataset
    if isinstance(dataset.processor.tokenizer, CachedTokenizer):
        return dict(dataset._text_runtime_metadata)
    started = time.monotonic()
    tokenizer = dataset.processor.tokenizer
    texts = list(dict.fromkeys(
        f"{dataset.text_instruction}{record['text']}" for record in dataset.records
    ))
    encodings = {}
    for start in range(0, len(texts), chunk_size):
        part = texts[start:start + chunk_size]
        encoded = tokenizer(part, padding=False, truncation=True,
                            max_length=dataset.max_text_length)
        for index, text in enumerate(part):
            encodings[text] = {key: tuple(value[index]) for key, value in encoded.items()}
    processor = copy.copy(dataset.processor)
    processor.tokenizer = CachedTokenizer(tokenizer, encodings, dataset.max_text_length)
    dataset.processor = processor
    metadata = dict(
        version=VERSION, module="mmfnd/input_runtime.py",
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=len(dataset.records), unique_texts=len(texts),
        encoded_tokens=sum(len(row["input_ids"]) for row in encodings.values()),
        preparation_seconds=time.monotonic() - started,
    )
    dataset._text_runtime_metadata = metadata
    dataset._text_runtime_contract = dict(text_instruction=dataset.text_instruction,
                                         max_text_length=dataset.max_text_length)
    return dict(metadata)


def cached_token_lengths(dataset) -> list[int] | None:
    """Reuse unpadded cache lengths for the exact current replay dataset.

    None means the caller must run the original length-tokenization path. In
    particular, an unknown/mutated tokenizer, truncation setting, prompt, or
    missing text must never silently supply stale lengths to the sampler.
    Invalid cached lengths and incompatible replay indexing fail as they do in
    the original sampler helper; no tokenization or dataset __getitem__ occurs.
    """
    processor = getattr(dataset, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    records = getattr(dataset, "records", None)
    if not isinstance(tokenizer, CachedTokenizer) or records is None:
        return None
    contract = getattr(dataset, "_text_runtime_contract", None)
    current = dict(text_instruction=getattr(dataset, "text_instruction", None),
                   max_text_length=getattr(dataset, "max_text_length", None))
    if (contract != current or current["max_text_length"] != tokenizer.max_length
            or tokenizer._signature() != tokenizer.signature):
        return None
    lengths = []
    for record in records:
        text = f"{current['text_instruction']}{record['text']}"
        encoded = tokenizer.encodings.get(text)
        if encoded is None:
            return None
        ids = encoded.get("input_ids")
        if not isinstance(ids, (list, tuple)):
            raise ValueError("cached token row requires input_ids")
        lengths.append(len(ids))
    if lengths and len(dataset) != len(lengths):
        if len(dataset) % len(lengths):
            raise RuntimeError(
                f"token length count mismatch: {len(lengths)} source rows for "
                f"{len(dataset)} dataset items"
            )
        repeat = len(dataset) // len(lengths)
        lengths = [value for value in lengths for _ in range(repeat)]
    if len(lengths) != len(dataset):
        raise RuntimeError(f"token length count mismatch: {len(lengths)} != {len(dataset)}")
    if any(value <= 0 or value > int(tokenizer.max_length) for value in lengths):
        raise ValueError("token lengths must be within the configured truncation limit")
    return lengths
