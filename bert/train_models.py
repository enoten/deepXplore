'''
Fine-tune the three ticket classifiers used by DeepXplore.

Mirrors https://github.com/enoten/support_ticket_analysis/blob/main/finetune_bert_classifier.ipynb
for Model1 (bert-base-uncased), and trains Model2/Model3 the same way on sibling BERT models.

usage:
  python train_models.py                 # train all three
  python train_models.py --model Model1  # train one
  python train_models.py --epochs 5 --quick
'''

import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    logging,
)

from configs import ID2LABEL, LABEL2ID, MODEL_SPECS, TEXT_FIELD, bcolors
from utils import CHECKPOINT_ROOT, checkpoint_dir, load_support_ticket_splits

logging.set_verbosity_error()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_classes = np.argmax(predictions, axis=1)
    acc = float((pred_classes == labels).mean())
    return {'Accuracy': round(acc, 3)}


def train_one(name, epochs=15, batch_size=8, lr=1e-4, seed=0, device=None):
    """Fine-tune one classifier and save to checkpoints/<name>/."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    hub_id = MODEL_SPECS[name]
    print(bcolors.HEADER + f'Training {name} from {hub_id} on device={device}' + bcolors.ENDC)

    # All three checkpoints are WordPiece BERT-family; share Model1's tokenizer so
    # smaller hubs with incomplete tokenizer files (e.g. bert-mini) still train.
    tok_src = MODEL_SPECS['Model1'] if name != 'Model1' else hub_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        hub_id,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    splits = load_support_ticket_splits(seed=seed)

    def preprocess(examples):
        return tokenizer(examples[TEXT_FIELD], truncation=True)

    tokenized = splits.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    out_dir = checkpoint_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    # Notebook used lr=1e-2 and 100 epochs; that is aggressive for bert-base.
    # Defaults here are more stable; pass --epochs/--lr to match the notebook.
    args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_ROOT, f'{name}_runs'),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='Accuracy',
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        seed=seed + hash(name) % 1000,
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # transformers API renamed tokenizer -> processing_class
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate(tokenized['test'])
    print(bcolors.OKGREEN + f'{name} test metrics: {metrics}' + bcolors.ENDC)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(bcolors.OKBLUE + f'Saved {name} -> {out_dir}' + bcolors.ENDC)
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT ticket classifiers for DeepXplore')
    parser.add_argument('--model', choices=list(MODEL_SPECS) + ['all'], default='all')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true',
                        help='fast smoke train: 3 epochs (good for checking the pipeline)')
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3

    names = list(MODEL_SPECS) if args.model == 'all' else [args.model]
    for name in names:
        train_one(name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)


if __name__ == '__main__':
    main()
