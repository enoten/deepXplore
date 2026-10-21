'''
Visualize BERT DeepXplore results logged by gen_diff.py.

usage:
  python visualize.py
  python visualize.py generated_inputs/synonym
'''

import argparse
import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from configs import ID2LABEL

MODEL_NAMES = ['BERT-base', 'DistilBERT', 'BERT-small']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
STATUS_COLORS = {'generated': 'tab:green', 'already_differ': 'tab:olive', 'failed': 'lightgray'}


def find_runs(paths):
    files = sorted({f for p in paths for f in glob.glob(os.path.join(p, '**', 'results.json'), recursive=True)})
    runs = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            run = json.load(fh)
        run['dir'] = os.path.dirname(f)
        transformation = run['args']['transformation']
        folder = os.path.basename(run['dir'])
        run['name'] = transformation if folder == transformation else f'{transformation} [{folder}]'
        runs.append(run)
    return runs


def label_names(preds):
    return [ID2LABEL[int(p)] for p in preds]


def plot_examples(run, out_dir, n):
    gens = [r for r in run['records'] if r['status'] == 'generated'][:n]
    if not gens:
        return None
    fig, axes = plt.subplots(len(gens), 1, figsize=(10, 1.8 * len(gens) + 0.5), squeeze=False)
    for row, r in enumerate(gens):
        ax = axes[row, 0]
        ax.axis('off')
        orig = textwrap.fill(r['orig_text'], 90)
        gen = textwrap.fill(r['generated_text'], 90)
        body = (
            f"seed {r['seed_idx']} | {label_names(r['orig_predictions'])}  -->  "
            f"{label_names(r['predictions'])}  ({r['iterations']} iters)\n"
            f"ORIG: {orig}\n"
            f"GEN : {gen}"
        )
        ax.text(0.01, 0.5, body, va='center', ha='left', fontsize=8, family='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#cccccc'))
    fig.suptitle(f'Difference-inducing tickets: {run["name"]}\n'
                 f'predictions = {" / ".join(MODEL_NAMES)}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f'examples_{os.path.basename(run["dir"])}.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_coverage_growth(runs, out_dir):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        cov = np.array([r.get('coverage') or [0, 0, 0] for r in run['records']], dtype=float)
        if cov.ndim == 1:
            continue
        x = np.arange(1, len(cov) + 1)
        for k in range(min(3, cov.shape[1])):
            ax.step(x, cov[:, k], where='post', color=COLORS[k],
                    label=f'{MODEL_NAMES[k]} ({run["neurons_per_model"][k]} neurons)')
        weights = np.array(run['neurons_per_model'][:cov.shape[1]], dtype=float)
        ax.step(x, cov @ weights / weights.sum(), where='post', color='black', ls='--', label='averaged')
        hits = [i + 1 for i, r in enumerate(run['records']) if r['status'] != 'failed']
        ax.plot(hits, np.zeros(len(hits)) + 0.02, '|', color='tab:green', ms=10, label='input found')
        ax.set_title(f'{run["name"]}  (t={run["args"]["threshold"]:.2f})', fontsize=9)
        ax.set_xlabel('seeds processed')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel('neuron coverage')
    axes[0, -1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    path = os.path.join(out_dir, 'coverage_growth.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(runs, out_dir):
    names = [run['name'] for run in runs]
    y = np.arange(len(runs))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(runs) + 2.4))

    ax = axes[0]
    left = np.zeros(len(runs))
    for status, color in STATUS_COLORS.items():
        counts = np.array([sum(r['status'] == status for r in run['records']) for run in runs])
        ax.barh(y, counts, left=left, color=color, label=status)
        for yi, (l, c) in enumerate(zip(left, counts)):
            if c:
                ax.text(l + c / 2, yi, str(c), ha='center', va='center', fontsize=8)
        left += counts
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('seeds')
    ax.set_title('Outcome per seed')
    ax.legend(fontsize=7)

    ax = axes[1]
    for yi, run in enumerate(runs):
        iters = [r['iterations'] for r in run['records'] if r['status'] == 'generated']
        if iters:
            ax.boxplot(iters, positions=[yi], orientation='horizontal', widths=0.5, showfliers=False)
            jitter = np.random.default_rng(yi).uniform(-0.15, 0.15, len(iters))
            ax.scatter(iters, yi + jitter, s=12, color='tab:green', alpha=0.7, zorder=3)
    ax.set_yticks(y, names)
    ax.set_ylim(len(runs) - 0.5, -0.5)
    ax.set_xlabel('iterations until models disagree')
    ax.set_title('Search cost')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'summary.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', default=['./generated_inputs'])
    parser.add_argument('-o', '--out_dir', default='./figures')
    parser.add_argument('-n', '--n_examples', type=int, default=6)
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        raise SystemExit(f'no results.json found under {args.paths} — run gen_diff.py first')
    os.makedirs(args.out_dir, exist_ok=True)
    paths = [plot_examples(run, args.out_dir, args.n_examples) for run in runs]
    paths += [plot_coverage_growth(runs, args.out_dir), plot_summary(runs, args.out_dir)]
    print('wrote:\n  ' + '\n  '.join(p for p in paths if p))
