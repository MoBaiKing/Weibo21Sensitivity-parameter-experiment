#!/usr/bin/env python3
"""Continue verified cross-fit folds without changing their scientific identity.

The original reference implementation remains the fingerprinted protocol oracle.
A partial best checkpoint has no optimizer/RNG state and cannot resume fitting;
only a calibrated checkpoint with matching complete OOF predictions is reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from mmfnd.dataset_contract import bind_dataset_workspace
from mmfnd.engine import autocast_context
from mmfnd.factory import build_loader, build_processor
from mmfnd.r1_cache import digest, file_sha, validate_probabilities
from mmfnd.reference_pipeline import (
    ReferenceModel, build_targets_from_oof, input_tensors, label_free_batches,
    prepare_identity,
)
from mmfnd.reference_subset_probe import smoothed_prior, subset_ce
from mmfnd.utils import cleanup_distributed, init_distributed, load_config, seed_everything


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_torch(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_rows(rows, expected_keys, *, fold=None, checkpoint_hash=None, availability_by_id=None):
    keys = [(str(row["sample_id"]), row["augmentation_id"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise ValueError("saved probability augmentation coverage mismatch")
    for row in rows:
        if fold is not None and (
            row.get("fold") != fold["fold"]
            or row.get("fit_id_hash") != fold["fit_id_hash"]
            or row.get("cal_id_hash") != fold["cal_id_hash"]
            or row.get("reference_checkpoint_sha256") != checkpoint_hash
        ):
            raise ValueError("saved OOF fold/checkpoint provenance mismatch")
        expected_availability = ([True] * 4 if availability_by_id is None
                                 else availability_by_id.get(str(row["sample_id"])))
        if expected_availability is None or row.get("availability") != expected_availability:
            raise ValueError("saved reference availability mismatch")
    validate_probabilities(torch.stack([row["q"] for row in rows]))


def verified_fold(directory, fingerprint, fold, prior, augmentation_ids, availability_by_id=None):
    """Return reusable artifacts, None for incomplete fitting, fail on corruption."""
    directory = Path(directory)
    checkpoint_path = directory / f"fold_{fold['fold']}.pth"
    oof_path = directory / f"oof_{fold['fold']}.pt"
    if not checkpoint_path.is_file():
        if oof_path.exists():
            raise ValueError("OOF predictions exist without their reference checkpoint")
        return None
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("fingerprint") != fingerprint or saved.get("fold") != fold:
        raise ValueError("paused reference checkpoint contract mismatch")
    if not torch.equal(saved["prior"].cpu(), prior.cpu()):
        raise ValueError("paused reference prior mismatch")
    if set(saved.get("state", {})) != set(saved.get("trainable_names", [])):
        raise ValueError("paused reference trainable key mismatch")
    temperature = saved.get("temperature")
    if temperature is None or not oof_path.is_file():
        return None
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("invalid saved reference temperature")
    payload = torch.load(oof_path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("paused OOF fingerprint mismatch")
    checkpoint_hash = file_sha(checkpoint_path)
    expected = {(str(sid), aid) for sid in fold["target_ids"] for aid in augmentation_ids}
    validate_rows(payload["rows"], expected, fold=fold, checkpoint_hash=checkpoint_hash,
                  availability_by_id=availability_by_id)
    return saved, payload["rows"], checkpoint_hash


def archive_incomplete_fold(directory, fold_number):
    paths = [directory / name for name in (
        f"fold_{fold_number}.pth", f"oof_{fold_number}.pt", f"val_{fold_number}.pt",
    )]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / "interrupted_folds" / f"fold_{fold_number}_{stamp}"
    destination.mkdir(parents=True)
    for path in existing:
        path.rename(destination / path.name)
    print(json.dumps({"reference_fold": fold_number, "restart_incomplete_only": str(destination)}), flush=True)


def fit_fold(model, dataset, fold, config, context, prior, fingerprint, directory):
    """Same seed, loss, optimizer, sampler, early stopping and temperature rule."""
    from train import build_optimizer
    from mmfnd.reference_runtime import calibration_logits, move_batch, partition_loader

    fit_loader = partition_loader(dataset, fold["fit_ids"], config, context, True)
    cal_loader = partition_loader(dataset, fold["cal_ids"], config, context)
    optimizer = build_optimizer(model, config)
    precision = config["train"]["precision"]
    accumulation = int(config["train"]["grad_accum_steps"])
    best, bad, fit_steps = float("inf"), 0, 0
    history, best_calibration = [], None
    train_seconds, calibration_seconds = 0., 0.
    checkpoint_path = directory / f"fold_{fold['fold']}.pth"
    for epoch in range(1, int(config["reference"].get("epochs", 3)) + 1):
        phase_start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(fit_loader.sampler, "set_epoch"):
            fit_loader.sampler.set_epoch(epoch)
        for step, raw in enumerate(fit_loader, 1):
            batch = move_batch(raw, context.device)
            window = min(accumulation, len(fit_loader) - ((step - 1) // accumulation) * accumulation)
            with autocast_context(context.device, precision):
                loss = subset_ce(model(input_tensors(batch)), batch["labels"], batch["availability"])
            if not torch.isfinite(loss):
                raise FloatingPointError("reference loss nonfinite")
            (loss / window).backward()
            if step % accumulation == 0 or step == len(fit_loader):
                norm = nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip_norm"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("reference gradient nonfinite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                fit_steps += 1
        train_seconds += time.monotonic() - phase_start
        phase_start = time.monotonic()
        logits, labels, availability = calibration_logits(model, cal_loader, context.device, precision)
        calibration_seconds += time.monotonic() - phase_start
        nll = float(subset_ce(logits, labels, availability))
        history.append(dict(epoch=epoch, cal_nll=nll))
        if nll < best:
            best, bad = nll, 0
            # Evaluation is deterministic and this is the exact best checkpoint's
            # calibration pass; no second encoder pass is needed for temperature.
            best_calibration = logits, labels, availability
            names = {name for name, param in model.named_parameters() if param.requires_grad}
            save_torch(dict(
                state={name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in names},
                trainable_names=sorted(names), fold=fold, prior=prior, epoch=epoch, fingerprint=fingerprint,
            ), checkpoint_path)
        else:
            bad += 1
        print(json.dumps(dict(reference_fold=fold["fold"], epoch=epoch, cal_nll=nll)), flush=True)
        if bad >= int(config["reference"].get("patience", 2)):
            break
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = model.state_dict()
    state.update(saved["state"])
    model.load_state_dict(state, strict=True)
    logits, labels, availability = best_calibration
    candidates = config["reference"].get("temperature_grid", [.5, 1., 2., 4.])
    temperature = min(candidates, key=lambda value: (
        float(subset_ce(logits / float(value), labels, availability)), abs(float(value) - 1),
    ))
    saved["temperature"] = temperature
    save_torch(saved, checkpoint_path)
    return saved, dict(optimizer_steps=fit_steps, history=history, calibration_encoder_pass_reused=True,
                       train_seconds=train_seconds, calibration_seconds=calibration_seconds)


def continue_reference(root, config, context, directory, *, verify_only=False, build_targets=False):
    if context.distributed:
        raise ValueError("reference continuation requires nproc=1")
    if (config["reference"].get("initialization") != "pretrained_only"
            or config["reference"].get("label_smoothing", 0) != 0
            or config["reference"].get("class_weight") is not None):
        raise ValueError("reference requires pretrained-only initialization and unweighted, unsmoothed CE")
    directory = Path(directory)
    started = time.monotonic()
    records, folds, fingerprint = prepare_identity(root, config)
    manifest_path = directory / "reference_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest and manifest.get("fingerprint") != fingerprint:
        raise ValueError("paused reference manifest fingerprint mismatch")
    if not manifest and directory.exists() and any(path.name != "preflight.json" for path in directory.iterdir()):
        raise ValueError("reference artifacts exist without their provenance manifest")
    labels = {str(record["id"]): record["label"] for record in records}
    processed = Path(config["data"]["processed_dir"])
    processed = processed if processed.is_absolute() else root / processed
    val_records = [json.loads(line) for line in (processed / "val.jsonl").read_text().splitlines() if line]
    from mmfnd.full_version_runtime import enabled, record_availability
    availability_by_id = None
    if enabled(config):
        all_records = records + val_records
        if len({str(record["id"]) for record in all_records}) != len(all_records):
            raise ValueError("train and validation reference IDs must be distinct")
        availability_by_id = {str(record["id"]): record_availability(record) for record in all_records}
    augmentation_ids = ("clean", "aug1") if config["data"].get("image_preprocessing", {}).get("train_augmentation", {}).get("enabled", False) else ("clean",)
    priors = [smoothed_prior([labels[sid] for sid in fold["fit_ids"]], float(config["reference"].get("prior_smoothing", 1))) for fold in folds["folds"]]
    completed = {}
    for fold, prior in zip(folds["folds"], priors):
        verified = verified_fold(directory, fingerprint, fold, prior, augmentation_ids, availability_by_id)
        if verified is not None:
            # Avoid retaining every trainable checkpoint in CPU memory.
            completed[fold["fold"]] = verified[2]
    report = dict(status="verified", fingerprint=digest(fingerprint), completed_folds=sorted(completed),
                  restart_folds=[fold["fold"] for fold in folds["folds"] if fold["fold"] not in completed])
    print(json.dumps(report), flush=True)
    if verify_only:
        return report

    # A cutoff between probability export and target construction needs no model.
    oof_path = directory / "oof_probabilities.pt"
    val_path = directory / "val_probabilities.pt"
    if len(completed) == len(folds["folds"]) and oof_path.is_file() and val_path.is_file():
        payload = torch.load(oof_path, map_location="cpu", weights_only=False)
        val_payload = torch.load(val_path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint or val_payload.get("fingerprint") != fingerprint:
            raise ValueError("combined reference probability fingerprint mismatch")
        validate_rows(payload["rows"], {(str(record["id"]), aid) for record in records for aid in augmentation_ids},
                      availability_by_id=availability_by_id)
        fold_map = {fold["fold"]: fold for fold in folds["folds"]}
        for number, fold in fold_map.items():
            rows = [row for row in payload["rows"] if row.get("fold") == number]
            validate_rows(rows, {(str(sid), aid) for sid in fold["target_ids"] for aid in augmentation_ids},
                          fold=fold, checkpoint_hash=completed[number], availability_by_id=availability_by_id)
        validate_rows(val_payload["rows"], {(str(record["id"]), "clean") for record in val_records},
                      availability_by_id=availability_by_id)
        if build_targets:
            build_targets_from_oof(directory, records)
        return report

    from mmfnd.reference_runtime import apply_model_runtime, partition_loader, predict_reference

    directory.mkdir(parents=True, exist_ok=True)
    save_json(dict(fingerprint=fingerprint, status="continuing", config=config, previous_costs=manifest.get("costs", [])), manifest_path)
    processor = build_processor(root, config)
    dataset = build_loader(root, config, "train", processor, shuffle=True).dataset
    actual_augmentations = ("clean", "aug1") if dataset.pool else ("clean",)
    if actual_augmentations != augmentation_ids:
        raise ValueError("configured reference replay pool differs from dataset")
    val_loader = build_loader(root, config, "val", processor)
    expected_val_keys = {(str(record["id"]), "clean") for record in val_loader.dataset.records}
    all_rows, val_folds, costs = [], [], []
    for fold, prior in zip(folds["folds"], priors):
        fold_start = time.monotonic()
        number = fold["fold"]
        checkpoint_path = directory / f"fold_{number}.pth"
        reused = number in completed
        rows, val_rows, model = None, None, None
        cost = dict(fold=number, reused_completed_fold=reused, optimizer_steps=0)
        if reused:
            saved, rows, checkpoint_hash = verified_fold(directory, fingerprint, fold, prior, augmentation_ids,
                                                        availability_by_id)
            val_path = directory / f"val_{number}.pt"
            if val_path.is_file():
                val_payload = torch.load(val_path, map_location="cpu", weights_only=False)
                if (val_payload.get("fingerprint") != fingerprint
                        or val_payload.get("checkpoint_sha256") != checkpoint_hash):
                    raise ValueError("saved validation fold provenance mismatch")
                val_rows = val_payload["rows"]
                validate_rows(val_rows, expected_val_keys, availability_by_id=availability_by_id)
        else:
            archive_incomplete_fold(directory, number)
        if not reused or val_rows is None:
            # Each fold has independent initialization/RNG, exactly as the oracle.
            seed_everything(int(config["seed"]))
            phase_start = time.monotonic()
            model = ReferenceModel(config).to(context.device)
            apply_model_runtime(model, config)
            cost["model_init_seconds"] = time.monotonic() - phase_start
            if reused:
                names = {name for name, param in model.named_parameters() if param.requires_grad}
                if set(saved["state"]) != names or set(saved["trainable_names"]) != names:
                    raise ValueError("saved reference trainable parameter contract mismatch")
                state = model.state_dict()
                state.update(saved["state"])
                model.load_state_dict(state, strict=True)
                del state
            else:
                saved, fit_cost = fit_fold(model, dataset, fold, config, context, prior, fingerprint, directory)
                cost.update(fit_cost)
                checkpoint_hash = file_sha(checkpoint_path)
                target_loader = partition_loader(dataset, fold["target_ids"], config, context)
                phase_start = time.monotonic()
                rows = predict_reference(model, label_free_batches(target_loader), context.device, config["train"]["precision"], prior, saved["temperature"])
                cost["oof_export_seconds"] = time.monotonic() - phase_start
                for row in rows:
                    row.update(fold=number, fit_id_hash=fold["fit_id_hash"], cal_id_hash=fold["cal_id_hash"], reference_checkpoint_sha256=checkpoint_hash)
                save_torch(dict(fingerprint=fingerprint, rows=rows), directory / f"oof_{number}.pt")
            phase_start = time.monotonic()
            val_rows = predict_reference(model, label_free_batches(val_loader), context.device, config["train"]["precision"], prior, saved["temperature"])
            cost["validation_export_seconds"] = time.monotonic() - phase_start
            save_torch(dict(fingerprint=fingerprint, checkpoint_sha256=checkpoint_hash, rows=val_rows, split="val", labels_read=False), directory / f"val_{number}.pt")
        all_rows.extend(rows)
        val_folds.append(val_rows)
        cost.update(seconds=time.monotonic() - fold_start, temperature=saved["temperature"], checkpoint_epoch=int(saved["epoch"]))
        costs.append(cost)
        save_json(dict(fingerprint=fingerprint, status="continuing", config=config, costs=costs,
                       previous_costs=manifest.get("costs", [])), manifest_path)
        del model, saved
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
    save_torch(dict(fingerprint=fingerprint, rows=all_rows), directory / "oof_probabilities.pt")
    val_rows = val_folds[0]
    for index, row in enumerate(val_rows):
        if any(values[index]["sample_id"] != row["sample_id"] for values in val_folds):
            raise ValueError("validation ensemble ID mismatch")
        row["q"] = torch.stack([values[index]["q"] for values in val_folds]).mean(0)
    save_torch(dict(fingerprint=fingerprint, rows=val_rows, split="val", labels_read=False), directory / "val_probabilities.pt")
    save_json(dict(fingerprint=fingerprint, status="probabilities_complete_continued", config=config, costs=costs,
                   previous_costs=manifest.get("costs", []), total_seconds=time.monotonic() - started,
                   runtime_sha256=file_sha(ROOT / "mmfnd/reference_runtime.py"),
                   continuation_sha256=file_sha(Path(__file__))), manifest_path)
    if build_targets:
        build_targets_from_oof(directory, records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["weibo21", "weibo", "gossipcop"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--build-targets", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed
    config["data"]["replay_seed"] = args.seed
    from mmfnd.full_version_runtime import installed
    with installed(config) as runtime:
        print(json.dumps({"full_version_runtime": runtime}), flush=True)
        bind_dataset_workspace(ROOT, config, args.dataset, config.get("dataset", {}).get("manifest_dir", f"datasets/{args.dataset}/ready"))
        context = init_distributed()
        try:
            continue_reference(ROOT, config, context, args.output, verify_only=args.verify_only, build_targets=args.build_targets)
        finally:
            cleanup_distributed()


if __name__ == "__main__":
    main()
