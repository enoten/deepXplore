'''
DeepXplore for BERT — continuous embedding ascent that turns AGREEMENT into DISAGREEMENT.

Goal
----
Only process seeds where all three models initially agree. Then run gradient ascent on
token embeddings (MNIST-style) until the projected text makes the ensemble disagree
(or retries are exhausted).

Update rule (continuous):
    embeds <- embeds + step * normalize(grad) * content_mask

If nearest-token projection does not change the string (stuck in the same vocab cell),
the step size is increased and a HotFlip discrete flip is applied on the highest-grad
token so the input actually leaves the seed token sequence.

usage:
  python gen_diff_embedding_grads.py light 1 0.1 2.0 5 40 0 -r 0
  python gen_diff_embedding_grads.py occl 1 0.1 2.0 5 40 0 -o ./generated_inputs/emb_occl
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
    constrain_embed_grad,
    content_position_mask,
    encode_texts,
    embeds_to_nearest_token_ids,
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

SUGGESTED_TEXT_PROJECTION = '''
1) Nearest vocab neighbor (IMPLEMENTED)
   t_i = argmin_v ||e_i - E[v]||, then decode.
2) HotFlip discrete flip when NN is stuck (IMPLEMENTED)
   Replace highest-grad token with v* = argmax_v grad_i · (E[v] - E[t_i]).
3) Constrained NN / LM rewrite — optional extensions for fluency.
'''

# ==============================================================================================
# Step 1: hyperparameters
# ==============================================================================================
parser = argparse.ArgumentParser(
    description='Embedding ascent: force agreement -> disagreement'
)
parser.add_argument(
    'transformation', choices=['light', 'occl', 'blackout'],
    help='embedding-space constraint (analogs of MNIST light / occl / blackout)',
)
parser.add_argument('weight_diff', type=float, help='lambda1: push target model away from label c')
parser.add_argument('weight_nc', type=float, help='lambda2: activate uncovered neurons')
parser.add_argument('step', type=float, help='initial step size s for embeds += s*grad')
parser.add_argument('seeds', type=int, help='number of agreeing seeds to process')
parser.add_argument('grad_iterations', type=int, help='max ascent iterations per seed')
parser.add_argument('threshold', type=float, help='neuron activation threshold t')
parser.add_argument('-t', '--target_model', choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-r', '--random_seed', type=int, default=None)
parser.add_argument('-o', '--out_dir', default='./generated_inputs/embedding_ascent')
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--occl_start', type=int, default=1)
parser.add_argument('--occl_width', type=int, default=4)
parser.add_argument('--project_every', type=int, default=1)
parser.add_argument('--max_draws', type=int, default=None,
                    help='max pool draws while searching for agreeing seeds')
parser.add_argument('--step_grow', type=float, default=1.5,
                    help='multiply step when projection leaves text unchanged')
parser.add_argument('--max_step', type=float, default=50.0,
                    help='cap on adaptive step size')
parser.add_argument('--diff_boost', type=float, default=2.0,
                    help='extra multiplier on lambda1 during ascent (stronger disagreement push)')
args = parser.parse_args()

if args.random_seed is not None:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(bcolors.HEADER + f'Device: {device}' + bcolors.ENDC)
print(bcolors.OKBLUE + 'Mode: only agreeing seeds; ascent until disagreement\n'
      + SUGGESTED_TEXT_PROJECTION + bcolors.ENDC)

seed_texts, _ = ticket_texts(split='test', seed=args.data_seed)
pool = seed_texts

tokenizer = shared_tokenizer()
models = [
    Model1(device=device),
    Model2(device=device),
    Model3(device=device),
]
ref_model = models[0]
ref_emb = ref_model.get_input_embeddings()
ref_weight = ref_emb.weight.detach()
ref_dim = ref_weight.size(1)

model3_dim = models[2].config.hidden_size
if model3_dim != ref_dim:
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    proj_3 = torch.randn(ref_dim, model3_dim, generator=gen, device=device)
    proj_3 = proj_3 / (proj_3.norm(dim=0, keepdim=True) + 1e-5)
else:
    proj_3 = None

model_layer_dicts = init_coverage_tables(models)
# Effective differential weight (stronger push toward disagreement)
eff_weight_diff = args.weight_diff * args.diff_boost


def record_and_report(text):
    for model, layer_dict in zip(models, model_layer_dicts):
        update_coverage(model, tokenizer, text, layer_dict, threshold=args.threshold, device=device)
    stats = [neuron_covered(d) for d in model_layer_dicts]
    print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
          % (stats[0][1], stats[0][2], stats[1][1], stats[1][2], stats[2][1], stats[2][2]) + bcolors.ENDC)
    averaged_nc = sum(s[0] for s in stats) / float(sum(s[1] for s in stats))
    print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
    return [s[2] for s in stats]


def _names(preds):
    return [ID2LABEL[int(p)] for p in preds]


def print_seed_vs_new(seed_idx, orig_text, gen_text, orig_preds, new_preds, status,
                      iterations=None, step_used=None, loss_val=None):
    print(bcolors.WARNING + '=' * 72 + bcolors.ENDC)
    print(bcolors.BOLD + f'seed {seed_idx} [{status}]' + bcolors.ENDC)
    if iterations is not None:
        extra = f'  step={step_used:.3f}' if step_used is not None else ''
        if loss_val is not None:
            extra += f'  loss={loss_val:.4f}'
        print(f'  ascent steps: {iterations}{extra}')
    print(bcolors.OKBLUE + '  --- ORIGINAL SEED (models AGREED) ---' + bcolors.ENDC)
    print(f'  text : {orig_text}')
    print(f'  preds: {_names(orig_preds)}  {list(orig_preds)}')
    print(bcolors.OKGREEN + '  --- NEW SAMPLE (after gradients) ---' + bcolors.ENDC)
    print(bcolors.OKGREEN + f'  text : {gen_text}' + bcolors.ENDC)
    print(bcolors.OKGREEN + f'  preds: {_names(new_preds)}  {list(new_preds)}' + bcolors.ENDC)
    print(f'  text changed: {"YES" if orig_text.strip() != (gen_text or "").strip() else "NO"}')
    disagrees = len(set(new_preds)) > 1
    msg = 'YES (agreement -> disagreement)' if disagrees else 'NO'
    color = bcolors.OKGREEN if disagrees else bcolors.FAIL
    print(color + f'  ensemble disagrees: {msg}' + bcolors.ENDC)
    print(bcolors.WARNING + '=' * 72 + bcolors.ENDC)


def project_to_text(embeds, attention_mask, seed_ids=None):
    ids = embeds_to_nearest_token_ids(
        embeds.detach(), ref_weight, attention_mask=attention_mask, forbid_ids=None,
    )
    if seed_ids is not None:
        special = set(tokenizer.all_special_ids)
        for i, tid in enumerate(seed_ids[0].tolist()):
            if tid in special:
                ids[0, i] = tid
    return ids, ids_to_text(tokenizer, ids, attention_mask)


def embeds_for_model(model, embeds_ref):
    h = model.config.hidden_size
    if h == embeds_ref.size(-1):
        return embeds_ref
    if model is models[2] and proj_3 is not None:
        return embeds_ref @ proj_3
    raise RuntimeError(f'no embedding map for hidden_size={h}')


def joint_loss_and_embed_grads(embeds, attention_mask, orig_label, neurons, weight_diff):
    """
    Differential + coverage objective. weight_diff is boosted so the target model is
    pushed off label c harder than the MNIST defaults (needed for discrete projection).
    """
    embeds = embeds.detach().requires_grad_(True)
    logits_list = []
    neuron_vals = []
    for model, (layer_idx, dim) in zip(models, neurons):
        e_m = embeds_for_model(model, embeds)
        logits, hidden = forward_logits(model, None, attention_mask, inputs_embeds=e_m)
        layer_idx = min(layer_idx, len(hidden) - 1)
        dim = min(dim, hidden[layer_idx].size(-1) - 1)
        neuron_vals.append(hidden[layer_idx][0, 0, dim])
        logits_list.append(logits)

    loss = joint_loss_from_logits(
        logits_list, orig_label, args.target_model, weight_diff, neuron_vals, args.weight_nc,
    )
    # Extra term: maximize spread of predicted-class logits across models (encourage disagreement)
    # on the current label c — detach-free, differentiable in embeds.
    logit_c = torch.stack([lg[0, orig_label] for lg in logits_list])
    loss = loss + 0.5 * ((logit_c - logit_c.mean()) ** 2).sum()

    loss.backward()
    grad = embeds.grad.detach()
    return normalize_grad(grad), float(loss.detach().item())


@torch.no_grad()
def predict_labels_from_embeds(embeds, attention_mask, seed_ids=None):
    ids, text = project_to_text(embeds, attention_mask, seed_ids=seed_ids)
    preds = predict_labels(models, tokenizer, text, device=device)
    return preds, text, ids


@torch.no_grad()
def hotflip_force_token_change(embeds, grad, seed_ids, attention_mask, content_mask):
    """
    When continuous NN projection is stuck, flip the highest-grad content token to the
    vocab neighbor that best aligns with the gradient (HotFlip), then reset embeds
    from the new discrete ids. Guarantees the token sequence changes.
    """
    # grad: [1, L, H], content_mask: [1, L, 1]
    norms = (grad * content_mask).norm(dim=-1)[0]  # [L]
    # ignore positions that are special (mask 0)
    if float(norms.max()) <= 0:
        return embeds, seed_ids, False

    pos = int(norms.argmax().item())
    g = grad[0, pos]  # [H]
    cur_id = int(seed_ids[0, pos].item())
    # score_v = g · (E[v] - E[cur])
    e_cur = ref_weight[cur_id]
    scores = (ref_weight - e_cur) @ g  # [V]
    # forbid specials and current token
    for tid in tokenizer.all_special_ids:
        scores[tid] = -1e9
    scores[cur_id] = -1e9
    new_id = int(scores.argmax().item())
    if new_id == cur_id:
        return embeds, seed_ids, False

    new_ids = seed_ids.clone()
    new_ids[0, pos] = new_id
    new_embeds = ref_emb(new_ids).detach()
    return new_embeds, new_ids, True


# ==============================================================================================
# Main loop: only agreeing seeds; ascent until disagreement
# ==============================================================================================
found = 0
records = []
skipped_already_differ = 0
draws = 0
max_draws = args.max_draws if args.max_draws is not None else max(50 * args.seeds, 100)
seed_idx = 0

print(bcolors.HEADER +
      f'Target: {args.seeds} agreeing seed(s) -> force disagreement '
      f'(max_draws={max_draws}, init_step={args.step}, diff_boost={args.diff_boost})'
      + bcolors.ENDC)

while seed_idx < args.seeds and draws < max_draws:
    draws += 1
    orig_text = random.choice(pool)
    labels = predict_labels(models, tokenizer, orig_text, device=device)

    if len(set(labels)) > 1:
        skipped_already_differ += 1
        print(bcolors.WARNING +
              f'draw {draws}: skip already-disagree {labels} '
              f'({_names(labels)})' + bcolors.ENDC)
        continue

    print(bcolors.OKGREEN +
          f'draw {draws}: agreeing seed {seed_idx} all={_names(labels)} {labels}'
          + bcolors.ENDC)
    print(f'  ORIGINAL SEED: {orig_text}')

    record = {
        'seed_idx': seed_idx,
        'status': 'failed',
        'iterations': None,
        'predictions': None,
        'orig_text': orig_text,
        'generated_text': None,
        'files': None,
        'mode': 'embedding_ascent_agree_to_disagree',
        'orig_predictions': labels,
    }
    records.append(record)

    orig_label = labels[0]
    neurons = [neuron_to_cover(d) for d in model_layer_dicts]
    batch = encode_texts(tokenizer, [orig_text], device=device)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    working_ids = input_ids.clone()

    embeds = ref_emb(working_ids).detach().clone()
    content_mask = content_position_mask(tokenizer, working_ids).to(device)
    cur_step = float(args.step)
    last_loss = None
    stalled = 0
    success = False
    text = orig_text
    preds = labels
    proj_ids = working_ids

    for iters in range(args.grad_iterations):
        grads, loss_val = joint_loss_and_embed_grads(
            embeds, attention_mask, orig_label, neurons, eff_weight_diff,
        )
        last_loss = loss_val
        grads = constrain_embed_grad(
            grads, args.transformation, content_mask,
            start=args.occl_start, width=args.occl_width,
        )

        embeds = embeds + grads * cur_step
        with torch.no_grad():
            base = ref_emb(working_ids)
            embeds = embeds * content_mask + base * (1.0 - content_mask)

        if (iters + 1) % args.project_every != 0 and iters + 1 < args.grad_iterations:
            continue

        preds, text, proj_ids = predict_labels_from_embeds(
            embeds, attention_mask, seed_ids=input_ids,
        )

        # If projection did not change tokens, grow step and HotFlip one token
        if text.strip() == orig_text.strip() or torch.equal(proj_ids, working_ids):
            stalled += 1
            cur_step = min(cur_step * args.step_grow, args.max_step)
            embeds, working_ids, flipped = hotflip_force_token_change(
                embeds, grads, working_ids, attention_mask, content_mask,
            )
            content_mask = content_position_mask(tokenizer, working_ids).to(device)
            preds, text, proj_ids = predict_labels_from_embeds(
                embeds, attention_mask, seed_ids=input_ids,
            )
            print(bcolors.WARNING +
                  f'  iter {iters + 1}: projection stalled -> step={cur_step:.3f}'
                  f'  hotflip={"yes" if flipped else "no"}'
                  + bcolors.ENDC)
        else:
            # snap continuous state onto the new discrete tokens (stable canvas)
            working_ids = proj_ids
            embeds = ref_emb(working_ids).detach().clone()
            content_mask = content_position_mask(tokenizer, working_ids).to(device)
            stalled = 0

        if len(set(preds)) > 1:
            print(bcolors.OKBLUE +
                  f'  SUCCESS: agreement -> disagreement after {iters + 1} iters '
                  f'(step={cur_step:.3f}, loss={loss_val:.3f})' + bcolors.ENDC)
            print_seed_vs_new(
                seed_idx, orig_text, text, labels, preds, 'generated',
                iterations=iters + 1, step_used=cur_step, loss_val=loss_val,
            )
            cov = record_and_report(text)
            prefix = f'{seed_idx:03d}_emb_{args.transformation}_{preds[0]}_{preds[1]}_{preds[2]}'
            payload = {
                'orig_text': orig_text,
                'generated_text': text,
                'orig_predictions': labels,
                'predictions': preds,
                'label_names': [ID2LABEL[p] for p in preds],
                'iterations': iters + 1,
                'transformation': args.transformation,
                'mode': 'embedding_ascent_agree_to_disagree',
                'step_final': cur_step,
                'step_init': args.step,
                'projection': 'nearest_vocab_neighbor+hotflip',
                'projected_token_ids': proj_ids[0].tolist(),
            }
            save_example(os.path.join(args.out_dir, prefix + '.json'), payload)
            save_example(os.path.join(args.out_dir, prefix + '_orig.json'), {
                'orig_text': orig_text,
                'predictions': labels,
                'label_names': [ID2LABEL[p] for p in labels],
            })
            record.update(
                status='generated', iterations=iters + 1, predictions=preds,
                generated_text=text, files=[prefix + '.json', prefix + '_orig.json'],
                coverage=cov, step_final=cur_step, last_loss=loss_val,
            )
            found += 1
            success = True
            break

    if not success:
        preds, text, proj_ids = predict_labels_from_embeds(
            embeds, attention_mask, seed_ids=input_ids,
        )
        record.update(
            status='failed', iterations=args.grad_iterations, predictions=preds,
            generated_text=text, coverage=[neuron_covered(d)[2] for d in model_layer_dicts],
            step_final=cur_step, last_loss=last_loss,
        )
        save_example(os.path.join(args.out_dir, f'{seed_idx:03d}_emb_{args.transformation}_failed.json'), {
            'orig_text': orig_text,
            'generated_text': text,
            'orig_predictions': labels,
            'predictions': preds,
            'status': 'failed',
            'iterations': args.grad_iterations,
            'step_final': cur_step,
        })
        print_seed_vs_new(
            seed_idx, orig_text, text, labels, preds, 'failed',
            iterations=args.grad_iterations, step_used=cur_step, loss_val=last_loss,
        )

    seed_idx += 1

if seed_idx < args.seeds:
    print(bcolors.FAIL +
          f'Stopped early: {seed_idx}/{args.seeds} agreeing seeds after {draws} draws '
          f'({skipped_already_differ} skipped). Raise --max_draws.'
          + bcolors.ENDC)

print(bcolors.HEADER +
      f'Done: {found}/{seed_idx} agreeing seeds converted to disagreement '
      f'(skipped {skipped_already_differ} already-disagree; draws={draws})'
      + bcolors.ENDC)

print()
print(bcolors.HEADER + '=== Summary: ORIGINAL (agree) vs NEW (after grads) ===' + bcolors.ENDC)
for r in records:
    print_seed_vs_new(
        r['seed_idx'], r['orig_text'], r.get('generated_text') or r['orig_text'],
        r['orig_predictions'], r.get('predictions') or r['orig_predictions'],
        r['status'], r.get('iterations'), r.get('step_final'), r.get('last_loss'),
    )

with open(os.path.join(args.out_dir, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'args': vars(args),
        'mode': 'embedding_ascent_agree_to_disagree',
        'eff_weight_diff': eff_weight_diff,
        'skipped_already_differ': skipped_already_differ,
        'pool_draws': draws,
        'agreeing_seeds_processed': seed_idx,
        'converted_to_disagreement': found,
        'neurons_per_model': [len(d) for d in model_layer_dicts],
        'id2label': ID2LABEL,
        'records': records,
    }, f, indent=1)

txt_path = os.path.join(args.out_dir, 'generated_from_seeds.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('Agreement -> disagreement via embedding gradient ascent\n')
    f.write('=' * 72 + '\n')
    for r in records:
        f.write(f"\nseed {r['seed_idx']} [{r['status']}]\n")
        f.write(f"  --- ORIGINAL SEED (agreed) ---\n")
        f.write(f"  text : {r['orig_text']}\n")
        f.write(f"  preds: {[ID2LABEL[p] for p in r['orig_predictions']]}\n")
        f.write(f"  --- NEW SAMPLE (after gradients) ---\n")
        f.write(f"  text : {r.get('generated_text')}\n")
        np_ = r.get('predictions') or r['orig_predictions']
        f.write(f"  preds: {[ID2LABEL[p] for p in np_]}\n")
        f.write(f"  iterations: {r.get('iterations')}  step_final: {r.get('step_final')}\n")
print(bcolors.OKBLUE + f'wrote text dump: {txt_path}' + bcolors.ENDC)
