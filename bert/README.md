# BERT DeepXplore

Generate difference-inducing support-ticket texts for three classifiers (bert-base, DistilBERT, and a 4-layer BERT). Labels are `access`, `disk`, and `user`. The three models are the oracle: a seed is difference-inducing when their predictions disagree.

Run commands from this directory (`bert_class/`).

## Setup

```powershell
pip install -r requirements.txt
```

Fine-tune the classifiers (writes checkpoints the loaders expect):

```powershell
python train_models.py
python train_models.py --model Model1
python Model1.py
python Model2.py
python Model3.py
```

`--quick` trains all three for 3 epochs. Other flags: `--epochs` (default 15), `--batch_size` (8), `--lr` (1e-4), `--seed` (0).

## Invoke `gen_diff.py`

Discrete, gradient-guided token edits. There is no step-size argument.

```text
python gen_diff.py <transformation> <weight_diff> <weight_nc> <seeds> <grad_iterations> <threshold>
```

| Position | Name | Meaning |
| --- | --- | --- |
| 1 | `transformation` | `synonym` (mild word swap), `mask` ([MASK] a high-grad token), or `typo` (character edit) |
| 2 | `weight_diff` | λ1, weight on differential behavior |
| 3 | `weight_nc` | λ2, weight on neuron coverage |
| 4 | `seeds` | number of test tickets to try |
| 5 | `grad_iterations` | max token-edit steps per seed |
| 6 | `threshold` | neuron activation threshold t |

Optional flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-t`, `--target_model` | `0` | model index (0, 1, or 2) pushed away from the agreed label |
| `-r`, `--random_seed` | unset | fix Python / PyTorch RNG |
| `-o`, `--out_dir` | `./generated_inputs` | where JSON examples and `results.json` are written |
| `--data_seed` | `0` | seed of the ticket train/test split |

```powershell
python gen_diff.py -h
python gen_diff.py synonym 1 0.1 10 10 0
python gen_diff.py mask 1 0.1 10 10 0 -t 0 -r 0
python gen_diff.py typo 1 0.1 10 10 0 -o .\generated_inputs\typo
```

## Invoke `gen_diff_embedding_grads.py`

Continuous embedding ascent, MNIST-style (`embeds += step * grad`), then nearest-token projection. Only seeds where all three models already agree are used. The step grows and a HotFlip token replacement is applied when projection leaves the text unchanged. Stops when the ensemble disagrees.

```text
python gen_diff_embedding_grads.py <transformation> <weight_diff> <weight_nc> <step> <seeds> <grad_iterations> <threshold>
```

| Position | Name | Meaning |
| --- | --- | --- |
| 1 | `transformation` | `light` (all content tokens), `occl` (a token window), or `blackout` (one highest-grad token) |
| 2 | `weight_diff` | λ1 |
| 3 | `weight_nc` | λ2 |
| 4 | `step` | initial ascent step size |
| 5 | `seeds` | number of **agreeing** seeds to convert (already-disagree draws are skipped) |
| 6 | `grad_iterations` | max ascent steps per seed |
| 7 | `threshold` | neuron activation threshold t |

Extra flags beyond `-t`, `-r`, `-o`, and `--data_seed`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--occl_start` | `1` | occlusion window start (token index) |
| `--occl_width` | `4` | occlusion window width |
| `--project_every` | `1` | project embeddings to tokens every N steps |
| `--max_draws` | `50 * seeds` | max pool draws while searching for agreeing seeds |
| `--step_grow` | `1.5` | multiply the step when the text does not change |
| `--max_step` | `50` | cap on the adaptive step |
| `--diff_boost` | `2.0` | extra multiplier on λ1 during ascent |

Default output directory: `./generated_inputs/embedding_ascent`.

```powershell
python gen_diff_embedding_grads.py -h
python gen_diff_embedding_grads.py light 1 0.1 2.0 5 40 0 -r 0 -o .\generated_inputs\emb_force_disagree
python gen_diff_embedding_grads.py occl 1 0.1 2.0 5 40 0 --occl_start 1 --occl_width 4 -o .\generated_inputs\emb_occl
```

`gen_diff_embedding_grads_only_new.py` is the same embedding ascent without the adaptive step, HotFlip, or `--diff_boost`. `seeds` is still the number of agreeing seeds. Default output: `./generated_inputs/embedding_ascent_only_new`.

```powershell
python gen_diff_embedding_grads_only_new.py light 1 0.1 0.5 5 20 0 -r 0 -o .\generated_inputs\emb_only_new
```

Each run writes per-seed JSON (generated text and, when edited, `*_orig.json`), `results.json`, and `generated_from_seeds.txt`.

## Print and visualize

```powershell
python print_inputs.py .\generated_inputs\synonym
python print_inputs.py .\generated_inputs\emb_force_disagree --generated-only
python visualize.py .\generated_inputs\synonym
```

`print_inputs.py` prints seed text versus the new text. `visualize.py` writes figures to `./figures` (`-o` to change the folder, `-n` for how many examples).
