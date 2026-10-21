# MNIST DeepXplore

Generate difference-inducing MNIST images for three LeNet models (LeNet-1, LeNet-4, LeNet-5). Gradient ascent changes a seed until the models disagree, under a domain constraint (`light`, `occl`, or `blackout`).

Run commands from this directory (`MNIST/`).

## Setup

```powershell
pip install -r requirements.txt
```

Weights load from `Model1.weights.h5` / `Model2.weights.h5` / `Model3.weights.h5` if present, otherwise the original `Model1.h5` / `Model2.h5` / `Model3.h5`. To retrain one model:

```powershell
python Model1.py
python Model2.py
python Model3.py
```

## Invoke `gen_diff.py`

```text
python gen_diff.py <transformation> <weight_diff> <weight_nc> <step> <seeds> <grad_iterations> <threshold>
```

| Position | Name | Meaning |
| --- | --- | --- |
| 1 | `transformation` | `light`, `occl`, or `blackout` |
| 2 | `weight_diff` | λ1, weight on differential behavior |
| 3 | `weight_nc` | λ2, weight on neuron coverage |
| 4 | `step` | gradient-ascent step size |
| 5 | `seeds` | number of test images to try |
| 6 | `grad_iterations` | max ascent steps per seed |
| 7 | `threshold` | neuron activation threshold t |

Optional flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-t`, `--target_model` | `0` | model index (0, 1, or 2) pushed away from the agreed label |
| `-sp`, `--start_point` | `0 0` | occlusion top-left `(row col)` |
| `-occl_size`, `--occlusion_size` | `10 10` | occlusion `(height width)` |
| `-r`, `--random_seed` | unset | fix Python / NumPy RNG |
| `-o`, `--out_dir` | `./generated_inputs` | where images and `results.json` are written |
| `--no_clip` | off | leave pixels outside `[0, 1]` |

Full help:

```powershell
python gen_diff.py -h
```

### Examples

Lighting constraint, 20 seeds, 50 ascent steps:

```powershell
python gen_diff.py light 1 0.1 10 20 50 0
```

Same run, reproducible, written to its own folder:

```powershell
python gen_diff.py light 1 0.1 10 20 50 0 -r 0 -o .\generated_inputs\light
```

Occlusion (10×10 patch starting at row 5, column 5):

```powershell
python gen_diff.py occl 1 0.1 10 20 50 0 -sp 5 5 -occl_size 10 10 -o .\generated_inputs\occl
```

Blackout (only darken pixels):

```powershell
python gen_diff.py blackout 1 0.1 10 20 50 0 -o .\generated_inputs\blackout
```

Each successful seed writes a generated image and the original (`*_orig.png`). A seed the models already disagree on is saved as `*_already_differ_*.png`. The run log is `results.json` in the output directory.

## Visualize a run

```powershell
python visualize.py .\generated_inputs\light
python visualize.py .\generated_inputs --compare
```

Figures go to `./figures`. See `python visualize.py -h`.
