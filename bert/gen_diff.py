'''
DeepXplore for BERT ticket classifiers.

Same joint objective as MNIST/gen_diff.py (paper Algorithm 1), adapted to text:
discrete gradient-guided token edits under domain constraints (synonym / mask / typo).

usage:
  python gen_diff.py synonym 1 0.1 10 10 0
  python gen_diff.py mask 1 0.1 10 10 0 -t 0 -r 0
  python gen_diff.py typo 1 0.1 10 10 0 -o ./generated_inputs/typo
'''

import argparse
import json
import os
import random

import torch

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import ID2LABEL, bcolors
from utils import (
    apply_constraint,
    encode_texts,
    forward_logits,
    ids_to_text,
    init_coverage_tables,
    joint_loss_from_logits,
    neuron_covered,
    neuron_to_cover,
    normalize_grad,
    predict_labels,
    save_example,
    shared_tokenizer,
    ticket_texts,
    update_coverage,
)

# ==============================================================================================
# Step 1: hyperparameters
# ==============================================================================================
parser = argparse.ArgumentParser(description='Difference-inducing input generation for BERT ticket models')
parser.add_argument('transformation', choices=['synonym', 'mask', 'typo'],
                    help='text domain constraint (analogs of light / occl / blackout)')
parser.add_argument('weight_diff', type=float, help='lambda1: push target model away from label c')
parser.add_argument('weight_nc', type=float, help='lambda2: activate uncovered neurons')
parser.add_argument('seeds', type=int, help='number of seed tickets')
parser.add_argument('grad_iterations', type=int, help='max token-edit iterations per seed')
parser.add_argument('threshold', type=float, help='neuron activation threshold t')
parser.add_argument('-t', '--target_model', choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-r', '--random_seed', type=int, default=None)
parser.add_argument('-o', '--out_dir', default='./generated_inputs')
parser.add_argument('--data_seed', type=int, default=0, help='seed used for the notebook data split')
args = parser.parse_args()

if args.random_seed is not None:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(bcolors.HEADER + f'Device: {device}' + bcolors.ENDC)

# ==============================================================================================
# Step 2: seed texts (labels unused as oracle — the three models are the oracle)
# ==============================================================================================
seed_texts, _ = ticket_texts(split='test', seed=args.data_seed)
if args.seeds > len(seed_texts):
    # allow sampling with replacement from the tiny nano test split
    pool = seed_texts
else:
    pool = seed_texts

# ==============================================================================================
# Step 3: three classifiers with similar functionality
# ==============================================================================================
tokenizer = shared_tokenizer()
models = [
    Model1(device=device),
    Model2(device=device),
    Model3(device=device),
]

# ==============================================================================================
# Step 4: neuron coverage tables (CLS dims across layers)
# ==============================================================================================
model_layer_dicts = init_coverage_tables(models)


def record_and_report(text):
    for model, layer_dict in zip(models, model_layer_dicts):
        update_coverage(model, tokenizer, text, layer_dict, threshold=args.threshold, device=device)
    stats = [neuron_covered(d) for d in model_layer_dicts]
    print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
          % (stats[0][1], stats[0][2], stats[1][1], stats[1][2], stats[2][1], stats[2][2]) + bcolors.ENDC)
    averaged_nc = sum(s[0] for s in stats) / float(sum(s[1] for s in stats))
    print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
    return [s[2] for s in stats]


def print_seed_vs_generated(seed_idx, orig_text, gen_text, orig_preds, new_preds, status, iterations=None):
    """Print the seed ticket and the new difference-inducing text side by side."""
    def names(preds):
        return [ID2LABEL[int(p)] for p in preds]

    print(bcolors.WARNING + '-' * 72 + bcolors.ENDC)
    print(bcolors.BOLD + f'seed {seed_idx} [{status}]' + bcolors.ENDC)
    print(f'  SEED text : {orig_text}')
    print(f'  SEED preds: {names(orig_preds)}  {list(orig_preds)}')
    if status == 'generated':
        print(bcolors.OKGREEN + f'  NEW  text : {gen_text}' + bcolors.ENDC)
        print(bcolors.OKGREEN + f'  NEW  preds: {names(new_preds)}  {list(new_preds)}'
              + (f'  (after {iterations} iters)' if iterations is not None else '') + bcolors.ENDC)
        if orig_text != gen_text:
            print(f'  changed   : yes')
        else:
            print(f'  changed   : no (same string after decode)')
    else:
        print(f'  NEW  text : (same as seed — models already disagree)')
        print(f'  NEW  preds: {names(new_preds)}  {list(new_preds)}')
    print(bcolors.WARNING + '-' * 72 + bcolors.ENDC)


def embedding_grads(input_ids, attention_mask, orig_label, neurons):
    """
    Compute d(obj)/d(embeds) for each model (HotFlip / DeepXplore gradient signal).
    Because the three models have different hidden sizes, we cannot share one embedding
    tensor the way MNIST shares pixels — we edit discrete tokens guided by these grads.
    """
    grads = []
    logits_list = []
    neuron_vals = []
    for model, (layer_idx, dim) in zip(models, neurons):
        emb_layer = model.get_input_embeddings()
        embeds = emb_layer(input_ids).detach().requires_grad_(True)
        logits, hidden = forward_logits(model, input_ids, attention_mask, inputs_embeds=embeds)
        # neuron activation: CLS hidden dim at the chosen layer
        neuron_act = hidden[layer_idx][0, 0, dim]
        logits_list.append(logits)
        neuron_vals.append(neuron_act)
        # per-model contribution is folded into joint loss below; store embeds for autograd
        embeds.retain_grad()
        # temporary store on model for the joint backward
        model._dx_embeds = embeds

    loss = joint_loss_from_logits(
        logits_list, orig_label, args.target_model, args.weight_diff, neuron_vals, args.weight_nc
    )
    for model in models:
        if model._dx_embeds.grad is not None:
            model._dx_embeds.grad.zero_()
    loss.backward()

    for model in models:
        g = model._dx_embeds.grad.detach()
        grads.append(normalize_grad(g))
        del model._dx_embeds
    return grads, float(loss.detach().item())


# ==============================================================================================
# Step 5: main loop over seeds
# ==============================================================================================
found = 0
records = []
for seed_idx in range(args.seeds):
    orig_text = random.choice(pool)
    text = orig_text
    record = {
        'seed_idx': seed_idx,
        'status': 'failed',
        'iterations': None,
        'predictions': None,
        'orig_text': orig_text,
        'generated_text': None,
        'files': None,
    }
    records.append(record)

    labels = predict_labels(models, tokenizer, text, device=device)
    record['orig_predictions'] = labels
    if len(set(labels)) > 1:
        print(bcolors.OKGREEN + f'input already causes different outputs: {labels}' + bcolors.ENDC)
        print_seed_vs_generated(seed_idx, orig_text, text, labels, labels, 'already_differ', iterations=0)
        cov = record_and_report(text)
        fname = f'{seed_idx:03d}_already_differ_{labels[0]}_{labels[1]}_{labels[2]}.json'
        save_example(os.path.join(args.out_dir, fname), {
            'orig_text': orig_text,
            'generated_text': text,
            'predictions': labels,
            'label_names': [ID2LABEL[p] for p in labels],
        })
        record.update(status='already_differ', iterations=0, predictions=labels,
                      generated_text=text, files=[fname], coverage=cov)
        found += 1
        continue

    orig_label = labels[0]
    neurons = [neuron_to_cover(d) for d in model_layer_dicts]
    batch = encode_texts(tokenizer, [text], device=device)
    input_ids = batch['input_ids'].clone()
    attention_mask = batch['attention_mask']

    for iters in range(args.grad_iterations):
        grads, loss_val = embedding_grads(input_ids, attention_mask, orig_label, neurons)
        new_ids = apply_constraint(args.transformation, tokenizer, input_ids, grads)
        if new_ids is None or torch.equal(new_ids, input_ids):
            break
        input_ids = new_ids
        text = ids_to_text(tokenizer, input_ids, attention_mask)

        preds = predict_labels(models, tokenizer, text, device=device)
        if len(set(preds)) > 1:
            print(bcolors.OKBLUE +
                  f'seed {seed_idx} ({ID2LABEL[orig_label]}) -> '
                  f'{[ID2LABEL[p] for p in preds]} after {iters + 1} iterations '
                  f'(loss={loss_val:.3f})' + bcolors.ENDC)
            print_seed_vs_generated(seed_idx, orig_text, text, labels, preds, 'generated',
                                    iterations=iters + 1)
            cov = record_and_report(text)
            prefix = f'{seed_idx:03d}_{args.transformation}_{preds[0]}_{preds[1]}_{preds[2]}'
            payload = {
                'orig_text': orig_text,
                'generated_text': text,
                'orig_predictions': labels,
                'predictions': preds,
                'label_names': [ID2LABEL[p] for p in preds],
                'iterations': iters + 1,
                'transformation': args.transformation,
            }
            save_example(os.path.join(args.out_dir, prefix + '.json'), payload)
            save_example(os.path.join(args.out_dir, prefix + '_orig.json'), {
                'orig_text': orig_text,
                'predictions': labels,
                'label_names': [ID2LABEL[p] for p in labels],
            })
            record.update(status='generated', iterations=iters + 1, predictions=preds,
                          generated_text=text, files=[prefix + '.json', prefix + '_orig.json'],
                          coverage=cov)
            found += 1
            break
    if record.get('coverage') is None:
        record['coverage'] = [neuron_covered(d)[2] for d in model_layer_dicts]

print(bcolors.HEADER + f'Done: {found} / {args.seeds} seeds produced difference-inducing inputs' + bcolors.ENDC)

# ==============================================================================================
# Step 6: print a summary of every new generated input from its seed
# ==============================================================================================
print()
print(bcolors.HEADER + '=== Generated inputs from seeds ===' + bcolors.ENDC)
n_shown = 0
for r in records:
    if r['status'] not in ('generated', 'already_differ'):
        continue
    n_shown += 1
    print_seed_vs_generated(
        r['seed_idx'], r['orig_text'], r.get('generated_text') or r['orig_text'],
        r['orig_predictions'], r['predictions'], r['status'], r.get('iterations'),
    )
if n_shown == 0:
    print('(none)')

with open(os.path.join(args.out_dir, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'args': vars(args),
        'neurons_per_model': [len(d) for d in model_layer_dicts],
        'id2label': ID2LABEL,
        'records': records,
    }, f, indent=1)

# also write a plain-text dump for quick reading
txt_path = os.path.join(args.out_dir, 'generated_from_seeds.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('Generated inputs from seeds\n')
    f.write('=' * 72 + '\n')
    for r in records:
        if r['status'] not in ('generated', 'already_differ'):
            continue
        f.write(f"\nseed {r['seed_idx']} [{r['status']}]\n")
        f.write(f"  SEED text : {r['orig_text']}\n")
        f.write(f"  SEED preds: {[ID2LABEL[p] for p in r['orig_predictions']]}\n")
        f.write(f"  NEW  text : {r.get('generated_text') or r['orig_text']}\n")
        f.write(f"  NEW  preds: {[ID2LABEL[p] for p in r['predictions']]}\n")
        if r.get('iterations') is not None:
            f.write(f"  iterations: {r['iterations']}\n")
print(bcolors.OKBLUE + f'wrote text dump: {txt_path}' + bcolors.ENDC)
