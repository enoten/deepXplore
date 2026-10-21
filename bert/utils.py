import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs import (
    DATASET_CONFIG,
    DATASET_NAME,
    ID2LABEL,
    KEY_PHRASE_TO_LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_SPECS,
    TEXT_FIELD,
    bcolors,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(HERE, 'checkpoints')

# Small synonym table for ticket-domain words (text analog of domain constraints).
SYNONYMS = {
    'access': ['permission', 'privileges', 'rights', 'entry'],
    'revoke': ['remove', 'cancel', 'withdraw', 'disable'],
    'grant': ['give', 'provide', 'allow', 'enable'],
    'disk': ['drive', 'storage', 'harddrive', 'volume'],
    'space': ['capacity', 'room', 'storage'],
    'full': ['filled', 'maxed', 'overflowing'],
    'error': ['failure', 'fault', 'issue', 'problem'],
    'user': ['account', 'employee', 'person', 'login'],
    'create': ['add', 'make', 'open', 'setup'],
    'delete': ['remove', 'erase', 'drop', 'purge'],
    'modify': ['update', 'change', 'edit', 'alter'],
    'profile': ['settings', 'account', 'role'],
    'please': ['kindly', 'urgently'],
    'need': ['require', 'want', 'request'],
    'help': ['assist', 'support', 'aid'],
    'immediate': ['urgent', 'asap', 'prompt'],
    'account': ['profile', 'login', 'user'],
    'permissions': ['access', 'rights', 'privileges'],
}


# ----------------------------------------------------------------------------------------------
# Data helpers (notebook pipeline)
# ----------------------------------------------------------------------------------------------

def dataset_to_df(dataset_obj, label=None):
    import pandas as pd
    cols = list(dataset_obj[0].keys())
    cols_values = {item: [] for item in cols}
    for rec in dataset_obj:
        for item in cols:
            cols_values[item].append(rec[item])
    if label is not None:
        cols_values['label'] = [label[rec['key_phrase']] for rec in dataset_obj]
    return pd.DataFrame(cols_values)


def load_support_ticket_splits(seed=0):
    """Load nano support tickets and recreate the notebook's 40/10/10 split."""
    raw = load_dataset(DATASET_NAME, name=DATASET_CONFIG)
    df = dataset_to_df(raw['train'], label=KEY_PHRASE_TO_LABEL)
    rng = np.random.default_rng(seed)
    random_index = rng.permutation(df.index.to_numpy())
    split_ind = np.array([[0, 40], [40, 50], [50, 60]])
    subsets = ['train', 'valid', 'test']
    ind2subset = {key: random_index[split_ind[i][0]:split_ind[i][1]] for i, key in enumerate(subsets)}
    split_dict = {key: Dataset.from_pandas(df.iloc[ind2subset[key]]) for key in subsets}
    return DatasetDict(split_dict)


def ticket_texts(split='test', seed=0):
    data = load_support_ticket_splits(seed=seed)[split]
    return list(data[TEXT_FIELD]), list(data['label'])


# ----------------------------------------------------------------------------------------------
# Model load / save
# ----------------------------------------------------------------------------------------------

def checkpoint_dir(name):
    return os.path.join(CHECKPOINT_ROOT, name)


def shared_tokenizer():
    """All three models are WordPiece BERT-family; use Model1's tokenizer as the shared vocab."""
    path = checkpoint_dir('Model1')
    src = path if os.path.isdir(path) else MODEL_SPECS['Model1']
    return AutoTokenizer.from_pretrained(src)


def load_classifier(name, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    path = checkpoint_dir(name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found. Run: python train_models.py  (or python train_models.py --model {name})'
        )
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    print(bcolors.OKBLUE + f'{name} loaded from {path}' + bcolors.ENDC)
    return model


def encode_texts(tokenizer, texts, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_label(model, tokenizer, text, device=None):
    batch = encode_texts(tokenizer, [text], device=device)
    logits = model(**batch).logits
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def predict_labels(models, tokenizer, text, device=None):
    return [predict_label(m, tokenizer, text, device=device) for m in models]


def forward_logits(model, input_ids, attention_mask, inputs_embeds=None):
    if inputs_embeds is None:
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    else:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
    return out.logits, out.hidden_states


# ----------------------------------------------------------------------------------------------
# Neuron coverage (CLS hidden dims across transformer layers)
# ----------------------------------------------------------------------------------------------

def coverage_layer_indices(model):
    """Use every transformer block's hidden state (plus embeddings = hidden_states[0])."""
    # hidden_states length = n_layers + 1; skip embeddings, keep encoder layers
    n = model.config.num_hidden_layers
    return list(range(1, n + 1))


def init_coverage_table(model):
    table = defaultdict(bool)
    hidden = model.config.hidden_size
    for layer_idx in coverage_layer_indices(model):
        for dim in range(hidden):
            table[(layer_idx, dim)] = False
    return table


def init_coverage_tables(models):
    return [init_coverage_table(m) for m in models]


def neuron_to_cover(model_layer_dict):
    not_covered = [k for k, v in model_layer_dict.items() if not v]
    if not_covered:
        return random.choice(not_covered)
    return random.choice(list(model_layer_dict.keys()))


def neuron_covered(model_layer_dict):
    covered = sum(1 for v in model_layer_dict.values() if v)
    total = len(model_layer_dict)
    return covered, total, covered / float(total)


def scale(x, rmax=1.0, rmin=0.0):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo == 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * (rmax - rmin) + rmin


@torch.no_grad()
def update_coverage(model, tokenizer, text, model_layer_dict, threshold=0.0, device=None):
    batch = encode_texts(tokenizer, [text], device=device)
    out = model(**batch, output_hidden_states=True)
    for layer_idx in coverage_layer_indices(model):
        # CLS token activation vector
        cls = out.hidden_states[layer_idx][0, 0].detach().cpu().numpy()
        scaled = scale(cls)
        for dim, val in enumerate(scaled):
            if val > threshold:
                model_layer_dict[(layer_idx, dim)] = True


def normalize_grad(g):
    return g / (g.norm() + 1e-5)


# ----------------------------------------------------------------------------------------------
# Continuous embedding space ↔ discrete tokens (for embedding-ascent DeepXplore)
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def embeds_to_nearest_token_ids(embeds, emb_weight, attention_mask=None, forbid_ids=None):
    """
    Project continuous token vectors onto the model's vocabulary embedding table.

    For each position i: token_i = argmin_v ||embeds[i] - E[v]||_2
    (equivalently argmax_v  embeds[i]·E[v] - ½||E[v]||² when comparing Euclidean distance).

    This is the standard way to turn a perturbed embedding sequence back into WordPiece IDs
    so the result can be decoded to readable text (HotFlip / embedding-space attacks).
    """
    # embeds: [B, L, H], emb_weight: [V, H]
    # argmin_v ||e - w_v||^2  <=>  argmax_v (e·w_v - 0.5||w_v||^2)
    ew = emb_weight
    half_norm = 0.5 * (ew * ew).sum(dim=-1)  # [V]
    scores = embeds @ ew.t() - half_norm      # [B, L, V]
    if forbid_ids:
        scores[..., list(forbid_ids)] = -1e9
    token_ids = scores.argmax(dim=-1)
    if attention_mask is not None:
        pad_id = 0
        token_ids = token_ids.masked_fill(attention_mask == 0, pad_id)
    return token_ids


@torch.no_grad()
def content_position_mask(tokenizer, input_ids):
    """Boolean mask [B, L]: True where continuous updates are allowed (non-special tokens)."""
    special = set(tokenizer.all_special_ids)
    ids = input_ids
    mask = torch.ones_like(ids, dtype=torch.float32)
    for b in range(ids.size(0)):
        for i, tid in enumerate(ids[b].tolist()):
            if tid in special:
                mask[b, i] = 0.0
    return mask.unsqueeze(-1)  # [B, L, 1] for broadcasting over H


def constrain_embed_grad(grad, name, content_mask, start=0, width=None):
    """
    Text-domain analogs of MNIST light / occl / blackout, applied in embedding space.
      light  — update all content positions (mild global shift)
      occl   — update only a contiguous window of content positions
      blackout — update only the single highest-|grad| content position
    """
    g = grad * content_mask
    if name == 'light':
        return g
    # flatten content scores
    B, L, H = g.shape
    if name == 'blackout':
        norms = g.norm(dim=-1)  # [B, L]
        # zero everything except argmax content token
        out = torch.zeros_like(g)
        for b in range(B):
            idx = int(norms[b].argmax().item())
            out[b, idx] = g[b, idx]
        return out
    if name == 'occl':
        out = torch.zeros_like(g)
        w = width if width is not None else max(1, L // 4)
        s = max(0, min(start, L - 1))
        out[:, s:s + w, :] = g[:, s:s + w, :]
        return out * content_mask
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------
# Domain-specific text constraints (analogs of light / occl / blackout)
# ----------------------------------------------------------------------------------------------

def _content_positions(tokenizer, input_ids):
    special = set(tokenizer.all_special_ids)
    ids = input_ids[0].tolist()
    return [i for i, tid in enumerate(ids) if tid not in special]


def constraint_mask(tokenizer, input_ids, grads_per_model, k=1):
    """Occlusion analog: replace the highest-gradient content token(s) with [MASK]."""
    pos = _content_positions(tokenizer, input_ids)
    if not pos:
        return None
    scores = []
    for p in pos:
        s = 0.0
        for g in grads_per_model:
            s += float(g[0, p].norm().item())
        scores.append((s, p))
    scores.sort(reverse=True)
    new_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.unk_token_id
    for _, p in scores[:k]:
        new_ids[0, p] = mask_id
    return new_ids


def constraint_typo(tokenizer, input_ids, grads_per_model, k=1):
    """Blackout analog: introduce a small character typo in a high-importance token."""
    pos = _content_positions(tokenizer, input_ids)
    if not pos:
        return None
    scores = [(float(sum(g[0, p].norm().item() for g in grads_per_model)), p) for p in pos]
    scores.sort(reverse=True)
    new_ids = input_ids.clone()
    for _, p in scores[:k]:
        token = tokenizer.convert_ids_to_tokens(int(new_ids[0, p]))
        word = token[2:] if token.startswith('##') else token
        if len(word) < 2 or not word.isalpha():
            continue
        chars = list(word)
        i = random.randrange(len(chars))
        # swap with neighbor or replace with nearby letter
        if i + 1 < len(chars) and random.random() < 0.5:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            chars[i] = random.choice(alphabet)
        new_word = ''.join(chars)
        # Retokenize just this piece; keep first subtoken id if multi-piece
        pieces = tokenizer.encode(new_word, add_special_tokens=False)
        if pieces:
            new_ids[0, p] = int(pieces[0])
        break
    return new_ids


def constraint_synonym(tokenizer, input_ids, grads_per_model, k=1):
    """Lighting-style mild semantic change: swap a content word with a synonym."""
    pos = _content_positions(tokenizer, input_ids)
    if not pos:
        return None
    scores = [(float(sum(g[0, p].norm().item() for g in grads_per_model)), p) for p in pos]
    scores.sort(reverse=True)
    new_ids = input_ids.clone()
    for _, p in scores[: max(3, k) * 3]:
        token = tokenizer.convert_ids_to_tokens(int(new_ids[0, p]))
        word = token[2:] if token.startswith('##') else token
        key = word.lower()
        if key not in SYNONYMS:
            continue
        cand = random.choice(SYNONYMS[key])
        pieces = tokenizer.encode(cand, add_special_tokens=False)
        if pieces:
            new_ids[0, p] = int(pieces[0])
            return new_ids
    # fallback: mask if no synonym found
    return constraint_mask(tokenizer, input_ids, grads_per_model, k=k)


def apply_constraint(name, tokenizer, input_ids, grads_per_model):
    if name == 'synonym':
        return constraint_synonym(tokenizer, input_ids, grads_per_model)
    if name == 'mask':
        return constraint_mask(tokenizer, input_ids, grads_per_model)
    if name == 'typo':
        return constraint_typo(tokenizer, input_ids, grads_per_model)
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------
# Joint objective helpers
# ----------------------------------------------------------------------------------------------

def joint_loss_from_logits(logits_list, orig_label, target_model, weight_diff, neuron_acts, weight_nc):
    """
    obj = sum_{k != j} F_k(x)[c] - lambda1 * F_j(x)[c] + lambda2 * sum neuron activations
    (same structure as MNIST gen_diff.py / paper Algorithm 1)
    """
    loss_diff = 0.0
    for k, logits in enumerate(logits_list):
        logit_c = logits[0, orig_label]
        loss_diff = loss_diff + (-weight_diff * logit_c if k == target_model else logit_c)
    loss_neuron = sum(neuron_acts)
    return loss_diff + weight_nc * loss_neuron


def ids_to_text(tokenizer, input_ids, attention_mask=None):
    if attention_mask is None:
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)
    length = int(attention_mask[0].sum().item())
    return tokenizer.decode(input_ids[0, :length], skip_special_tokens=True)


def save_example(path, payload):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
