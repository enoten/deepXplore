'''
Model1: google-bert/bert-base-uncased  (same LLM as the notebook)

usage: python Model1.py
'''

from train_models import train_one
from utils import load_classifier


def Model1(train=False, device=None):
    if train:
        return train_one('Model1', device=device)
    return load_classifier('Model1', device=device)


if __name__ == '__main__':
    Model1(train=True)
