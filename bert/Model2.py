'''
Model2: distilbert-base-uncased  (lighter BERT-family sibling for differential testing)

usage: python Model2.py
'''

from train_models import train_one
from utils import load_classifier


def Model2(train=False, device=None):
    if train:
        return train_one('Model2', device=device)
    return load_classifier('Model2', device=device)


if __name__ == '__main__':
    Model2(train=True)
