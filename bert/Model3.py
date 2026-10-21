'''
Model3: google/bert_uncased_L-4_H-256_A-4  (small BERT; same WordPiece vocab family)

usage: python Model3.py
'''

from train_models import train_one
from utils import load_classifier


def Model3(train=False, device=None):
    if train:
        return train_one('Model3', device=device)
    return load_classifier('Model3', device=device)


if __name__ == '__main__':
    Model3(train=True)
