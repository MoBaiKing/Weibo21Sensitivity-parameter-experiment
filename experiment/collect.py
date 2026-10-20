"""Collect finished TEST results: <out>/results.csv (one row per task), <out>/matrix.md / matrix.csv
(seed mean +- sample std of the 3x3 lambda_relation x lambda_utility grid and the tau row) and a small
<out>/results_bundle.tar.gz (metrics, per-sample predictions, frozen selections, histories, configs)."""
import csv
import json
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['setting', 'tau', 'lambda_relation', 'lambda_utility', 'seed', 'status', 'test_samples',
          'test_accuracy_pct', 'test_macro_f1_pct', 'test_auc_pct', 'fake_f1_pct', 'real_f1_pct', 'nll', 'brier',
          'ece', 'threshold', 'val_best_macro_f1_pct', 'epochs_run', 'gpu', 'targets_sha256', 'metrics_sha256',
          'predictions_sha256']
METRICS = [('test_macro_f1_pct', 'Macro-F1'), ('test_accuracy_pct', 'Accuracy'), ('test_auc_pct', 'AUC')]


def rows_of(out, status):
    grid = json.loads((ROOT / 'configs' / 'grid.json').read_text())
    rows, bundle = [], []
    for setting in grid['settings']:
        for seed in grid['seeds']:
            run = out / 'runs' / setting['name'] / f'seed{seed}'
            test = run / 'evaluation' / 'final' / 'test'
            row = dict(setting=setting['name'], tau=setting['tau'], lambda_relation=setting['lambda_relation'],
                       lambda_utility=setting['lambda_utility'], seed=seed,
                       status=status.get(f"{setting['name']}/seed{seed}", {}).get('status', 'NOT_RUN'))
            if (test / 'receipt.json').is_file():
                m = json.loads((test / 'metrics.json').read_text())
                receipt = json.loads((test / 'receipt.json').read_text())
                frozen = json.loads((run / 'frozen.json').read_text())
                history = json.loads((run / 'history.json').read_text()) if (run / 'history.json').is_file() else []
                row.update(status='TESTED', test_samples=m['samples'], test_accuracy_pct=100 * m['accuracy'],
                           test_macro_f1_pct=100 * m['macro_f1'], test_auc_pct=100 * m['auc'],
                           fake_f1_pct=100 * m['fake_f1'], real_f1_pct=100 * m['real_f1'], nll=m['nll'],
                           brier=m['brier'], ece=m['ece'], threshold=m['threshold'],
                           val_best_macro_f1_pct=100 * m['val_best_macro_f1'], epochs_run=len(history),
                           gpu=receipt.get('gpu'), targets_sha256=frozen['targets_sha256'],
                           metrics_sha256=receipt['metrics_sha256'], predictions_sha256=receipt['predictions_sha256'])
                for name in ('evaluation/final/test/metrics.json', 'evaluation/final/test/predictions.jsonl',
                             'evaluation/final/test/receipt.json', 'frozen.json', 'final_summary.json',
                             'history.json', 'config.json'):
                    if (run / name).is_file():
                        bundle.append(run / name)
            rows.append(row)
    return grid, rows, bundle


def cell(rows, setting, key):
    values = [r[key] for r in rows if r['setting'] == setting and r['status'] == 'TESTED']
    if not values:
        return '—', 0
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{statistics.mean(values):.2f} ± {std:.2f}', len(values)


def matrix(grid, rows):
    by = {(s['tau'], s['lambda_relation'], s['lambda_utility']): s['name'] for s in grid['settings']}
    rels, utils, taus = (0.1, 0.2, 0.4), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)
    lines, table = [], []
    for key, label in METRICS:
        lines += [f'### {label}（TEST，5 个种子均值 ± 样本标准差，括号内为已完成种子数）', '',
                  'λrel ＼ λutil | ' + ' | '.join(str(u) for u in utils), '|'.join(['---'] * (len(utils) + 1))]
        for rel in rels:
            cells = []
            for util in utils:
                text, n = cell(rows, by[(1.0, rel, util)], key)
                cells.append(f'{text} ({n})')
                table.append(dict(metric=label, block='lambda_grid', tau=1.0, lambda_relation=rel, lambda_utility=util,
                                  value=text, seeds=n))
            lines.append(f'**{rel}** | ' + ' | '.join(cells))
        lines += ['', 'τ（λrel=0.2，λutil=1.0） | ' + ' | '.join(str(t) for t in taus), '|'.join(['---'] * (len(taus) + 1))]
        cells = []
        for tau in taus:
            text, n = cell(rows, by[(tau, 0.2, 1.0)], key)
            cells.append(f'{text} ({n})')
            table.append(dict(metric=label, block='tau_row', tau=tau, lambda_relation=0.2, lambda_utility=1.0,
                              value=text, seeds=n))
        lines += [f'**{label}** | ' + ' | '.join(cells), '']
    return '\n'.join(lines) + '\n', table


def collect(out, status=None):
    out = Path(out).resolve()
    grid, rows, bundle = rows_of(out, status or {})
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text, table = matrix(grid, rows)
    (out / 'matrix.md').write_text(text)
    with (out / 'matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with tarfile.open(out / 'results_bundle.tar.gz', 'w:gz') as tar:
        for name in ('results.csv', 'matrix.md', 'matrix.csv', 'status.json', 'environment.json'):
            if (out / name).is_file():
                tar.add(out / name, arcname=name)
        for path in bundle:
            tar.add(path, arcname=str(path.relative_to(out)))
    return sum(r['status'] == 'TESTED' for r in rows), len(rows)


if __name__ == '__main__':
    import sys
    done, total = collect(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'outputs')
    print(f'tested {done}/{total}')
