"""One task of the Weibo21full_version sensitivity experiment (started by ../train.py, one process per task).

Stages (identical to the A10 runs of the IRAUL 7B Full model):
  reference  fresh 3-fold reference of one seed  -> <out>/references/seed<S>/targets.pt
  train      student of one (setting, seed) with that seed's reference -> validation-selected top-k average
  test       the frozen model once on the 913 TEST samples with its validation threshold
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / 'iraul'
sys.path.insert(0, str(CODE))
sys.path.insert(1, str(HERE))
os.chdir(CODE)

# Fixed 8:1:1 split shipped in Weibo21full_version.zip (splits_811/, image paths relative to the dataset dir).
SPLIT_SHA256 = {'train': '8bea8e1a2b9b2d7b00514bcf81d28c48bb10f8d64e8ebec9db196e93bb9474a4',
                'val': '837c26460b438a7ab291e5c4a7bd0f662354a4c1f4c3ff059ee35b17df924afd',
                'test': 'cffe2a9a35702cf5a233a63e82d2cbb14cf0f9b83d2e3e8ec3e19c0b4f90b64a'}
MICRO_BATCH, GRAD_ACCUM = 2, 8   # A10 7B Full: micro-batch 2 x accumulation 8 = effective batch 16
TEST_SIZE = 913


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'immutable artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def setting_of(name):
    grid = json.loads((ROOT / 'configs' / 'grid.json').read_text())
    return next(s for s in grid['settings'] if s['name'] == name)


def runtime_config(kind, seed, args, setting=None):
    """Template (A10 seed config) + this machine's paths; a setting changes only tau / lambda_relation / lambda_utility."""
    config = json.loads((ROOT / 'configs' / 'templates' / f'{kind}.json').read_text())
    data, models, out = Path(args.data).resolve(), Path(args.models).resolve(), Path(args.out).resolve()
    splits = data / 'splits_811'
    for split, expected in SPLIT_SHA256.items():
        if sha(splits / f'{split}.jsonl') != expected:
            raise ValueError(f'{splits / split}.jsonl is not the registered 8:1:1 split')
    config['seed'] = config['data']['replay_seed'] = int(seed)
    config['data']['root'] = config['data']['image_root'] = str(data)
    config['data']['processed_dir'] = config['dataset']['manifest_dir'] = str(splits)
    config['model']['text_backbone'] = str(models / 'Qwen2.5-7B-Instruct')
    config['model']['vision_backbone'] = str(models / 'siglip-base-patch16-224')
    if args.num_workers is not None:
        config['data']['num_workers'] = int(args.num_workers)   # not part of any fingerprint
    if kind == 'reference':
        config['train']['output_dir'] = str(out / 'references' / f'seed{seed}' / 'unused_outputs')
    else:
        config['model']['r1']['tau'] = float(setting['tau'])
        config['loss']['lambda_relation'] = float(setting['lambda_relation'])
        config['loss']['lambda_utility'] = float(setting['lambda_utility'])
        config['train']['output_dir'] = str(out / 'runs' / setting['name'])
    path = out / 'configs' / (f'reference_seed{seed}.json' if kind == 'reference' else f"{setting['name']}_seed{seed}.json")
    write(path, config)
    return config, path


def install_sha_cache(out):
    """Memoise file sha256 by (path, inode, size, mtime): avoids re-hashing the 16 GB backbone for every task."""
    import mmfnd.r1_cache as cache
    store = Path(out) / 'cache' / 'sha256.json'
    try:
        values = json.loads(store.read_text())
    except (OSError, ValueError):
        values = {}
    raw = cache.file_sha

    def file_sha(path):
        path = Path(path)
        stat = path.stat()
        key = f'{path.resolve()}|{stat.st_ino}|{stat.st_size}|{stat.st_mtime_ns}'
        if key not in values:
            values[key] = raw(path)
            store.parent.mkdir(parents=True, exist_ok=True)
            temporary = store.with_name(f'{store.name}.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(values))
            temporary.replace(store)
        return values[key]
    cache.file_sha = file_sha


def without_workers(config):
    config = deepcopy(config)
    config['data'].pop('num_workers', None)
    return config


def reference_dir(args, seed):
    return Path(args.out).resolve() / 'references' / f'seed{seed}'


def reference_done(directory):
    return all((directory / name).is_file() for name in ('targets.pt', 'targets.json', 'reference_manifest.json'))


def reference(args):
    seed = args.seed
    config, path = runtime_config('reference', seed, args)
    directory = reference_dir(args, seed)
    if reference_done(directory):
        print(json.dumps(dict(stage='reference', seed=seed, status='already_complete')), flush=True)
        return
    command = [sys.executable, '-B', str(CODE / 'scripts' / 'continue_masked_r1_reference.py'), '--dataset', 'weibo21',
               '--seed', str(seed), '--config', str(path), '--output', str(directory), '--build-targets']
    print(json.dumps(dict(stage='reference', seed=seed, started=datetime.now().astimezone().isoformat())), flush=True)
    code = subprocess.call(command, cwd=CODE)
    if code or not reference_done(directory):
        raise SystemExit(code or 1)
    print(json.dumps(dict(stage='reference', seed=seed, status='complete')), flush=True)


def run_dir(args, setting, seed):
    return Path(args.out).resolve() / 'runs' / setting / f'seed{seed}'


def train(args):
    from mmfnd.full_version_runtime import installed
    from mmfnd.training_progress import select_resume_checkpoint
    setting, seed = setting_of(args.setting), args.seed
    config, path = runtime_config('student', seed, args, setting)
    directory, run = reference_dir(args, seed), run_dir(args, args.setting, seed)
    if not reference_done(directory):
        raise ValueError(f'reference of seed {seed} is not complete')
    install_sha_cache(args.out)
    with installed(config):
        if (run / 'final_summary.json').is_file():
            return freeze(args, run, directory)
        saved = run / 'input_config.json'
        if saved.exists() and without_workers(json.loads(saved.read_text())) != without_workers(config):
            raise ValueError(f'{run} was started with different settings; use another --out or delete it')
        resume = select_resume_checkpoint(run / 'checkpoints')
        if resume is None and (run / 'checkpoints').exists() and any((run / 'checkpoints').iterdir()):
            raise ValueError('checkpoints exist without a valid resume point; inspect manually')
        if not saved.exists():
            write(saved, config)
        command = ['train.py', '--dataset', 'weibo21', '--seed', str(seed), '--config', str(path),
                   '--manifest-dir', config['data']['processed_dir'], '--run-name', run.name,
                   '--r1-target-cache', str(directory / 'targets.pt'),
                   '--per-gpu-batch-size', str(MICRO_BATCH), '--grad-accum-steps', str(GRAD_ACCUM)]
        if resume is not None:
            command += ['--resume', str(resume)]
        print(json.dumps(dict(stage='train', setting=args.setting, seed=seed, resume=str(resume) if resume else None,
                              started=datetime.now().astimezone().isoformat())), flush=True)
        import train as trainer
        previous, sys.argv = sys.argv, command
        try:
            trainer.main()
        finally:
            sys.argv = previous
        if not (run / 'final_summary.json').is_file():
            raise SystemExit(75)
        return freeze(args, run, directory)


def freeze(args, run, directory):
    import torch
    from mmfnd.evaluation import checkpoint_threshold_selection
    summary = json.loads((run / 'final_summary.json').read_text())
    checkpoint = run / 'checkpoints' / 'final_averaged.pth'
    if summary.get('final_model_kind') != 'top_k_average' or Path(summary['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('final summary is not the validation-selected top-k average')
    selection = checkpoint_threshold_selection(torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True))
    if selection.get('threshold_source', 'validation') != 'validation':
        raise ValueError('threshold must come from validation')
    frozen = dict(setting=args.setting, seed=args.seed, dataset='weibo21', data_version='Weibo21full_version',
                  split='8:1:1 (7302/913/913)', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  threshold_selection=selection, final_summary_sha256=sha(run / 'final_summary.json'),
                  runtime_config_sha256=sha(run / 'config.json'),
                  reference_manifest_sha256=sha(directory / 'reference_manifest.json'),
                  targets_sha256=sha(directory / 'targets.pt'), selection='validation_macro_f1_top_k_average')
    write(run / 'frozen.json', frozen, immutable=True)
    print(json.dumps(dict(stage='frozen', setting=args.setting, seed=args.seed, threshold=selection.get('threshold'))), flush=True)
    return frozen


def test(args):
    import torch
    from mmfnd.full_version_runtime import installed
    seed, run = args.seed, run_dir(args, args.setting, args.seed)
    frozen = json.loads((run / 'frozen.json').read_text())
    dest = run / 'evaluation' / 'final' / 'test'
    if (dest / 'receipt.json').is_file():
        print(json.dumps(dict(stage='test', setting=args.setting, seed=seed, status='already_tested')), flush=True)
        return
    if (dest / 'test_started.json').exists():
        raise ValueError('a TEST was started before; never re-expose TEST automatically (inspect manually)')
    actual = json.loads((run / 'config.json').read_text())
    if sha(frozen['checkpoint']) != frozen['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    with installed(actual):
        write(dest / 'test_started.json', dict(seed=seed, split='test', started_at=datetime.now().astimezone().isoformat(),
                                               checkpoint_sha256=frozen['checkpoint_sha256'],
                                               frozen_sha256=sha(run / 'frozen.json')), immutable=True)
        from mmfnd.engine import load_checkpoint, write_jsonl
        from mmfnd.evaluation import checkpoint_threshold_selection
        from mmfnd.evaluation_masked_r1 import evaluate_r1
        from mmfnd.factory import build_loader, build_model, build_processor
        from mmfnd.utils import seed_everything
        if actual['train'].get('r1_eval_runtime_optimization', False):
            from mmfnd.evaluation_runtime import install_prediction_runtime
            install_prediction_runtime()
        seed_everything(seed)
        device = torch.device('cuda')
        model = build_model(actual).to(device)
        checkpoint = load_checkpoint(Path(frozen['checkpoint']), model, device)
        if checkpoint_threshold_selection(checkpoint) != frozen['threshold_selection']:
            raise ValueError('loaded checkpoint changed the validation-frozen threshold')
        del checkpoint
        loader_config = deepcopy(actual)
        loader_config['train']['per_gpu_batch_size'] = 2
        loader = build_loader(CODE, loader_config, 'test', build_processor(CODE, actual))
        metrics, rows, _ = evaluate_r1(model, loader, device, 0, {0: 'fake', 1: 'real'},
                                      precision=actual['train']['precision'], split='test',
                                      threshold_selection=frozen['threshold_selection'])
        if len(rows) != TEST_SIZE:
            raise ValueError(f'8:1:1 TEST must have {TEST_SIZE} rows, got {len(rows)}')
        write(dest / 'metrics.json', metrics, immutable=True)
        write_jsonl(rows, dest / 'predictions.jsonl')
        write(dest / 'receipt.json', dict(split='test', checkpoint_sha256=frozen['checkpoint_sha256'],
                                          frozen_sha256=sha(run / 'frozen.json'), metrics_sha256=sha(dest / 'metrics.json'),
                                          predictions_sha256=sha(dest / 'predictions.jsonl'),
                                          gpu=torch.cuda.get_device_name(0)), immutable=True)
        print(json.dumps(dict(stage='test', setting=args.setting, seed=seed, accuracy=metrics['accuracy'],
                              macro_f1=metrics['macro_f1'], auc=metrics.get('auc'))), flush=True)
    if args.prune:
        # Keep final_averaged.pth (the tested model) and best.pth; drop resume-only and top-k files.
        checkpoints = run / 'checkpoints'
        for name in ('progress.pth', 'last.pth'):
            (checkpoints / name).unlink(missing_ok=True)
        shutil.rmtree(checkpoints / 'topk', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['reference', 'train', 'test'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--setting')
    parser.add_argument('--data', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--prune', action='store_true')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('GPU required: training/testing never runs on CPU')
    {'reference': reference, 'train': train, 'test': test}[args.stage](args)
