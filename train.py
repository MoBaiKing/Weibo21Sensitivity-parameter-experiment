#!/usr/bin/env python3
"""IRAUL - Weibo21full_version (8:1:1) parameter sensitivity, end to end on the local GPUs.

    python train.py --data /path/to/Weibo21full_version

1. checks the environment, the 8:1:1 split files and every image hash;
2. trains a fresh 3-fold reference for each of the 5 seeds (200404-200408);
3. trains every student of the grid (11 settings x 5 seeds) with its seed's reference,
   selects the validation top-k average and tests it once on the 913 TEST samples.
Tasks run in parallel (--procs-per-gpu processes on each GPU); a student starts as soon as its seed's
reference is ready. Re-running the same command resumes unfinished tasks and never re-tests a finished one.
Results: <out>/results.csv, <out>/matrix.md, <out>/results_bundle.tar.gz.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiment'))
import collect  # noqa: E402

PINNED = {'torch': '2.4.1+cu124', 'transformers': '4.46.3', 'peft': '0.13.2', 'Pillow': '12.3.0', 'tokenizers': '0.20.3'}
MODEL_FILES = {'Qwen2.5-7B-Instruct': ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
                                       'model-00001-of-00004.safetensors', 'model-00004-of-00004.safetensors'],
               'siglip-base-patch16-224': ['config.json', 'model.safetensors', 'preprocessor_config.json', 'spiece.model']}
stopping = False


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment(args):
    versions = {}
    for name in PINNED:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    wrong = {k: (v, PINNED[k]) for k, v in versions.items() if v != PINNED[k]}
    if wrong:
        message = f'package versions differ from the A10 experiment (installed, expected): {wrong}'
        if not args.allow_other_versions:
            raise SystemExit(message + '\nInstall requirements.txt (see README) or pass --allow-other-versions.')
        print('[warning] ' + message, flush=True)
    for folder, names in MODEL_FILES.items():
        missing = [n for n in names if not (Path(args.models) / folder / n).is_file()]
        if missing:
            raise SystemExit(f'model files missing in {Path(args.models) / folder}: {missing}; run scripts/download_models.sh')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU visible.')
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    write(args.out / 'environment.json', dict(checked_at=now(), python=sys.version, packages=versions, gpus=gpus,
                                              cuda=torch.version.cuda, cpu_count=os.cpu_count(),
                                              data=str(Path(args.data).resolve()), models=str(Path(args.models).resolve())))
    return len(gpus)


def check_data(data):
    """The fixed split files and every image they reference (sha256 recorded per image)."""
    splits = Path(data) / 'splits_811'
    if not (splits / 'dataset_manifest.json').is_file():
        raise SystemExit(f'{splits}/dataset_manifest.json not found: --data must be the unzipped Weibo21full_version directory')
    expected = {'train': '8bea8e1a2b9b2d7b00514bcf81d28c48bb10f8d64e8ebec9db196e93bb9474a4',
                'val': '837c26460b438a7ab291e5c4a7bd0f662354a4c1f4c3ff059ee35b17df924afd',
                'test': 'cffe2a9a35702cf5a233a63e82d2cbb14cf0f9b83d2e3e8ec3e19c0b4f90b64a'}
    checked, bad = 0, []
    for split, value in expected.items():
        path = splits / f'{split}.jsonl'
        if file_sha(path) != value:
            raise SystemExit(f'{path} differs from the published 8:1:1 split')
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            record = json.loads(line)
            for relative, digest in zip(record['images'], record['image_sha256']):
                image = Path(data) / relative
                if not image.is_file() or file_sha(image) != digest:
                    bad.append(str(image))
                checked += 1
    if bad:
        hint = ''
        if any(';' in Path(item).name for item in bad):
            hint = ("\nSome image names end with ';690'. The default unzip strips ';N' as a VMS version number; "
                    "re-extract with `unzip -V Weibo21full_version.zip -d <dir>` "
                    "or `python -m zipfile -e Weibo21full_version.zip <dir>`.")
        raise SystemExit(f'{len(bad)} images missing or changed, e.g. {bad[:3]}{hint}')
    print(f'[data] split files and {checked} images verified', flush=True)


def command(stage, seed, setting, args):
    cmd = [sys.executable, '-B', str(ROOT / 'experiment' / 'job.py'), '--stage', stage, '--seed', str(seed),
           '--data', str(Path(args.data).resolve()), '--models', str(Path(args.models).resolve()), '--out', str(args.out)]
    if args.num_workers is not None:
        cmd += ['--num-workers', str(args.num_workers)]
    if setting:
        cmd += ['--setting', setting]
    if stage == 'test' and not args.no_prune:
        cmd.append('--prune')
    return cmd


def environment(seed, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LEDGER_TRAIN_STOP_AT'):
        env.pop(key, None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='unzipped Weibo21full_version directory (contains splits_811/)')
    parser.add_argument('--models', default=str(ROOT / 'models'), help='directory with Qwen2.5-7B-Instruct/ and siglip-base-patch16-224/ (default ./models)')
    parser.add_argument('--out', default=str(ROOT / 'outputs'), help='output directory (default ./outputs)')
    parser.add_argument('--gpus', help='comma separated GPU ids (default: all visible)')
    parser.add_argument('--procs-per-gpu', type=int, default=3, help='parallel tasks per GPU (default 3; one task needs <= 24 GB)')
    parser.add_argument('--settings', default='all', help='comma separated setting names from configs/grid.json (default all 11)')
    parser.add_argument('--seeds', help='comma separated seeds (default: the 5 seeds of configs/grid.json)')
    parser.add_argument('--num-workers', type=int, help='DataLoader workers per task (default: 2, as in the A10 config)')
    parser.add_argument('--no-prune', action='store_true', help='keep resume and top-k checkpoints after TEST')
    parser.add_argument('--skip-data-check', action='store_true', help='skip the image hash check (already passed)')
    parser.add_argument('--allow-other-versions', action='store_true', help='only warn when package versions differ')
    parser.add_argument('--collect-only', action='store_true', help='only rebuild results.csv / matrix.md')
    args = parser.parse_args()
    args.out = Path(args.out).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = json.loads((ROOT / 'configs' / 'grid.json').read_text())
    names = [s['name'] for s in grid['settings']]
    settings = names if args.settings == 'all' else args.settings.split(',')
    unknown = set(settings) - set(names)
    if unknown:
        raise SystemExit(f'unknown settings: {sorted(unknown)}')
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else grid['seeds']
    status_path = args.out / 'status.json'
    status = json.loads(status_path.read_text())['tasks'] if status_path.exists() else {}
    if args.collect_only:
        print('tested %d / %d -> %s' % (*collect.collect(args.out, status), args.out / 'matrix.md'))
        return

    count = check_environment(args)
    gpus = [int(g) for g in args.gpus.split(',')] if args.gpus else list(range(count))
    slots = [g for g in gpus for _ in range(args.procs_per_gpu)]
    if not args.skip_data_check:
        check_data(args.data)

    def reference_done(seed):
        d = args.out / 'references' / f'seed{seed}'
        return all((d / n).is_file() for n in ('targets.pt', 'targets.json', 'reference_manifest.json'))

    def stage_of(setting, seed):
        run = args.out / 'runs' / setting / f'seed{seed}'
        if (run / 'evaluation/final/test/receipt.json').is_file():
            return None
        return 'test' if (run / 'frozen.json').is_file() else 'train'

    # Queue: references first, then students ordered by setting so partial results fill the matrix evenly.
    pending = [('reference', seed, None) for seed in seeds if not reference_done(seed)]
    pending += [(stage_of(s, seed), seed, s) for s in settings for seed in seeds if stage_of(s, seed)]
    for s in settings:
        for seed in seeds:
            if stage_of(s, seed) is None:
                status[f'{s}/seed{seed}'] = dict(status='TESTED')
    blocked = set()

    def publish():
        write(status_path, dict(updated_at=now(), out=str(args.out), slots=len(slots), tasks=status))

    def stop(signum, frame):
        global stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f'[run] {len(pending)} tasks to run on GPUs {gpus} ({len(slots)} parallel slots)', flush=True)

    running = {}
    while (pending or running) and not stopping:
        for index, gpu in enumerate(slots):
            if index in running:
                continue
            ready = next((t for t in pending if t[0] == 'reference' or (reference_done(t[1]) and t[1] not in blocked)), None)
            if ready is None:
                break
            pending.remove(ready)
            stage, seed, setting = ready
            key = f'reference/seed{seed}' if stage == 'reference' else f'{setting}/seed{seed}'
            log = (args.out / 'logs' / f"{key.replace('/', '_')}.{stage}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('ab')
            process = subprocess.Popen(command(stage, seed, setting, args), cwd=ROOT, env=environment(seed, gpu),
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            running[index] = (ready, key, process, stream, time.time())
            status[key] = dict(status='RUNNING', stage=stage, gpu=gpu, pid=process.pid, started=now(), log=str(log))
            publish()
            print(f'{now()} START {key} {stage} gpu{gpu}', flush=True)
        time.sleep(10)
        for index in list(running):
            (stage, seed, setting), key, process, stream, started = running[index]
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del running[index]
            hours = (time.time() - started) / 3600
            if code == 0 and stage == 'train':
                pending.insert(0, ('test', seed, setting))   # straight to its single TEST
                status[key] = dict(status='TRAINED', train_hours=round(hours, 2), finished=now())
            elif code == 0:
                status[key] = dict(status='TESTED' if stage == 'test' else 'REFERENCE_READY', hours=round(hours, 2), finished=now())
                if stage == 'test':
                    collect.collect(args.out, status)
            else:
                status[key] = dict(status='FAILED', stage=stage, returncode=code, finished=now(),
                                   log=str(args.out / 'logs' / f"{key.replace('/', '_')}.{stage}.log"))
                if stage == 'reference':
                    blocked.add(seed)
                    pending[:] = [t for t in pending if t[1] != seed]
            publish()
            print(f'{now()} EXIT {key} {stage} code={code} ({hours:.2f} h)', flush=True)
    if stopping:
        print('stopping: asking running tasks to exit; rerun the same command to resume', flush=True)
        for _, key, process, _, _ in running.values():
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.time() + 90
        for (stage, _, _), key, process, _, _ in running.values():
            try:
                process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            status[key] = dict(status='INTERRUPTED', stage=stage, finished=now())
        publish()
    done, total = collect.collect(args.out, status)
    print(f'[done] tested {done}/{total}; see {args.out / "matrix.md"}; send back {args.out / "results_bundle.tar.gz"}')


if __name__ == '__main__':
    main()
