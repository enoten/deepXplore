'''
Visualize the key DeepXplore results logged by gen_diff.py (results.json in each output directory).

usage: python visualize.py                                  # every run found under ./generated_inputs
       python visualize.py generated_inputs/occl            # a single run
       python visualize.py --compare                        # + neuron coverage comparison (paper Fig. 9)

Figures are written to ./figures:
  examples_<run>.png      original / generated / perturbation for difference-inducing inputs (paper Fig. 8)
  coverage_growth.png     neuron coverage per model as more seeds are processed
  summary.png             success rate, gradient-ascent iterations, and which model was fooled
  coverage_comparison.png neuron coverage vs threshold: DeepXplore vs adversarial vs random inputs
'''

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODEL_NAMES = ['LeNet-1', 'LeNet-4', 'LeNet-5']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
STATUS_COLORS = {'generated': 'tab:green', 'already_differ': 'tab:olive', 'failed': 'lightgray'}


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def find_runs(paths):
    files = sorted({f for p in paths for f in glob.glob(os.path.join(p, '**', 'results.json'), recursive=True)})
    runs = []
    for f in files:
        with open(f) as fh:
            run = json.load(fh)
        run['dir'] = os.path.dirname(f)
        transformation, folder = run['args']['transformation'], os.path.basename(run['dir'])
        run['name'] = transformation if folder == transformation else '%s [%s]' % (transformation, folder)
        runs.append(run)
    return runs


def load_img(run, fname):
    return np.asarray(Image.open(os.path.join(run['dir'], fname)), dtype='float32') / 255


def deviating_models(record):
    '''Indices of the models whose prediction differs from the original agreed label (or the majority).'''
    preds = record['predictions']
    if record['status'] == 'generated':
        ref = record['orig_predictions'][0]
    else:
        values, counts = np.unique(preds, return_counts=True)
        if counts.max() == 1:
            return list(range(len(preds)))
        ref = values[counts.argmax()]
    return [k for k, p in enumerate(preds) if p != ref]


# ----------------------------------------------------------------------------------------------
# Figures from the logs
# ----------------------------------------------------------------------------------------------

def plot_examples(run, out_dir, n):
    gens = [r for r in run['records'] if r['status'] == 'generated'][:n]
    if not gens:
        return None
    fig, axes = plt.subplots(3, len(gens), figsize=(1.7 * len(gens) + 0.8, 5.6), squeeze=False)
    for c, r in enumerate(gens):
        gen, orig = load_img(run, r['files'][0]), load_img(run, r['files'][1])
        diff = gen - orig
        fooled = deviating_models(r)

        axes[0, c].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, c].set_title('seed %d\nall predict %d' % (r['seed_idx'], r['orig_predictions'][0]), fontsize=8)
        axes[1, c].imshow(gen, cmap='gray', vmin=0, vmax=1)
        axes[1, c].set_title('%d / %d / %d\n%d iter.' % (*r['predictions'], r['iterations']), fontsize=8,
                             color=COLORS[fooled[0]] if len(fooled) == 1 else 'black')
        axes[2, c].imshow(diff, cmap='seismic', vmin=-1, vmax=1)
        axes[2, c].set_title('L1 = %.1f' % np.abs(diff).sum(), fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for row, label in enumerate(['original', 'generated', 'perturbation']):
        axes[row, 0].set_ylabel(label)
    fig.suptitle('Difference-inducing inputs: %s\npredictions = %s (title colored by the single fooled model)'
                 % (run['name'], ' / '.join(MODEL_NAMES)), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'examples_%s.png' % os.path.basename(run['dir']))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_coverage_growth(runs, out_dir):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs):
        cov = np.array([r['coverage'] for r in run['records']])
        x = np.arange(1, len(cov) + 1)
        for k in range(cov.shape[1]):
            ax.step(x, cov[:, k], where='post', color=COLORS[k],
                    label='%s (%d neurons)' % (MODEL_NAMES[k], run['neurons_per_model'][k]))
        weights = np.array(run['neurons_per_model'], dtype=float)
        ax.step(x, cov @ weights / weights.sum(), where='post', color='black', ls='--', label='averaged')
        hits = [i + 1 for i, r in enumerate(run['records']) if r['status'] != 'failed']
        ax.plot(hits, np.zeros(len(hits)) + 0.02, '|', color='tab:green', ms=10, label='input found')
        ax.set_title('%s  (threshold %.2f)' % (run['name'], run['args']['threshold']), fontsize=9)
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.6 * len(runs) + 2.6))

    # (a) outcome of every seed
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

    # (b) gradient-ascent iterations needed
    ax = axes[1]
    iters = [[r['iterations'] for r in run['records'] if r['status'] == 'generated'] for run in runs]
    for yi, it in enumerate(iters):
        if it:
            ax.boxplot(it, positions=[yi], orientation='horizontal', widths=0.5, showfliers=False)
            jitter = np.random.default_rng(yi).uniform(-0.15, 0.15, len(it))
            ax.scatter(it, yi + jitter, s=12, color='tab:green', alpha=0.7, zorder=3)
    ax.set_yticks(y, names)
    ax.set_ylim(len(runs) - 0.5, -0.5)
    ax.set_xlabel('iterations until models disagree')
    ax.set_title('Search cost')
    ax.grid(axis='x', alpha=0.3)

    # (c) which model gave the deviating answer
    ax = axes[2]
    height = 0.8 / len(MODEL_NAMES)
    for k, model_name in enumerate(MODEL_NAMES):
        counts = [sum(k in deviating_models(r) for r in run['records'] if r['status'] != 'failed') for run in runs]
        ax.barh(y - 0.4 + height * (k + 0.5), counts, height=height, color=COLORS[k], label=model_name)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('difference-inducing inputs')
    ax.set_title('Which model deviated')
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, 'summary.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------
# Neuron coverage comparison (paper Section 7.2 / Fig. 9): same number of inputs from each method
# ----------------------------------------------------------------------------------------------

def neuron_scores(model, x):
    '''Per neuron: max over inputs of the (per-layer min-max scaled, spatially averaged) activation.
    A neuron counts as covered at threshold t iff its score > t (same rule as utils.update_coverage).'''
    from utils import build_layer_extractor
    _, extractor = build_layer_extractor(model)
    scores = []
    for out in extractor(x, training=False):
        out = np.asarray(out)
        flat = out.reshape(len(out), -1)
        lo, hi = flat.min(1, keepdims=True), flat.max(1, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1)
        scaled = ((flat - lo) / rng).reshape(len(out), -1, out.shape[-1])
        scores.append(scaled.mean(1).max(0))
    return np.concatenate(scores)


def fgsm(model, x, y, eps):
    import keras
    import tensorflow as tf
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(keras.losses.sparse_categorical_crossentropy(y, model(x, training=False)))
    return np.clip(x.numpy() + eps * np.sign(tape.gradient(loss, x).numpy()), 0, 1)


def plot_coverage_comparison(runs, out_dir, thresholds, eps):
    from keras.layers import Input
    from Model1 import Model1
    from Model2 import Model2
    from Model3 import Model3
    from utils import load_mnist, IMG_ROWS, IMG_COLS

    generated = [load_img(run, r['files'][0]) for run in runs for r in run['records'] if r['files']]
    if not generated:
        return None
    gen_x = np.stack(generated)[..., None]
    n = len(gen_x)

    (_, _), (x_test, y_test) = load_mnist()
    idx = np.random.default_rng(0).choice(len(x_test), n, replace=False)
    rand_x, rand_y = x_test[idx], y_test[idx]

    input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, 1))
    models = [Model1(input_tensor=input_tensor), Model2(input_tensor=input_tensor), Model3(input_tensor=input_tensor)]

    methods = ['DeepXplore', 'adversarial (FGSM, eps=%.2f)' % eps, 'random test inputs']
    styles = [('tab:red', 'o', '-'), ('tab:purple', 's', '--'), ('tab:gray', '^', ':')]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True)
    print('\nNeuron coverage with %d inputs per method' % n)
    print('%-10s %-32s ' % ('model', 'method') + ' '.join('t=%-5.2f' % t for t in thresholds))
    for k, (ax, model) in enumerate(zip(axes, models)):
        sets = [gen_x, fgsm(model, rand_x, rand_y, eps), rand_x]
        for method, x, (color, marker, ls) in zip(methods, sets, styles):
            scores = neuron_scores(model, x)
            cov = [float(np.mean(scores > t)) for t in thresholds]
            ax.plot(thresholds, cov, color=color, marker=marker, ls=ls, label=method)
            print('%-10s %-32s ' % (MODEL_NAMES[k], method) + ' '.join('%-7.3f' % c for c in cov))
        ax.set_title(MODEL_NAMES[k])
        ax.set_xlabel('neuron activation threshold t')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('neuron coverage')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Neuron coverage of %d test inputs per method' % n, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, 'coverage_comparison.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize DeepXplore results logged by gen_diff.py')
    parser.add_argument('paths', nargs='*', default=['./generated_inputs'],
                        help="directories searched (recursively) for results.json")
    parser.add_argument('-o', '--out_dir', default='./figures')
    parser.add_argument('-n', '--n_examples', type=int, default=8, help="examples shown per run")
    parser.add_argument('--compare', action='store_true',
                        help="also compute neuron coverage of DeepXplore vs adversarial vs random inputs")
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75])
    parser.add_argument('--eps', type=float, default=0.1, help="FGSM perturbation size for --compare")
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        raise SystemExit('no results.json found under %s - run gen_diff.py first' % args.paths)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = [plot_examples(run, args.out_dir, args.n_examples) for run in runs]
    paths += [plot_coverage_growth(runs, args.out_dir), plot_summary(runs, args.out_dir)]
    if args.compare:
        paths.append(plot_coverage_comparison(runs, args.out_dir, args.thresholds, args.eps))
    print('\nwrote:\n  ' + '\n  '.join(p for p in paths if p))
