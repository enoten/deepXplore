'''
DeepXplore on MNIST: generate difference-inducing inputs for three LeNet variants.

usage: python gen_diff.py -h
example: python gen_diff.py light 1 0.1 10 50 20 0
'''

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import Input

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

# ==============================================================================================
# Step 1: hyperparameters
# ==============================================================================================
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior (lambda1)", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage (lambda2)", type=float)
parser.add_argument('step', help="step size of gradient ascent (s)", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient ascent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated (t)", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate (row col)",
                    default=(0, 0), type=int, nargs=2)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size (height width)",
                    default=(10, 10), type=int, nargs=2)
parser.add_argument('-r', '--random_seed', help="random seed for reproducibility", default=None, type=int)
parser.add_argument('-o', '--out_dir', help="where generated images are written", default='./generated_inputs')
parser.add_argument('--no_clip', help="do not keep pixels in [0, 1] (behaviour of the original repo)",
                    action='store_true')
args = parser.parse_args()

if args.random_seed is not None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
os.makedirs(args.out_dir, exist_ok=True)

# ==============================================================================================
# Step 2: unlabeled seed inputs (MNIST test set, labels are not needed - the models are the oracle)
# ==============================================================================================
(_, _), (x_test, _) = load_mnist()

# ==============================================================================================
# Step 3: three DNNs with similar functionality, sharing one input tensor
# ==============================================================================================
input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, 1))
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)
models = [model1, model2, model3]

# One forward pass per model returns every layer's output (logits, chosen neuron, prediction, coverage)
extractors = [build_layer_extractor(m) for m in models]

print(extractors)

exit()


# ==============================================================================================
# Step 4: neuron coverage tables
# ==============================================================================================
model_layer_dicts = list(init_coverage_tables(model1, model2, model3))


def predict_labels(img):
    return [int(np.argmax(m(img, training=False)[0])) for m in models]


def record_and_report(img):
    for (names, extractor), layer_dict in zip(extractors, model_layer_dicts):
        update_coverage(img, names, extractor, layer_dict, args.threshold)
    stats = [neuron_covered(d) for d in model_layer_dicts]
    print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
          % (stats[0][1], stats[0][2], stats[1][1], stats[1][2], stats[2][1], stats[2][2]) + bcolors.ENDC)
    averaged_nc = sum(s[0] for s in stats) / float(sum(s[1] for s in stats))
    print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)


def joint_loss_and_grads(gen_img, orig_label, neurons):
    '''
    obj = obj1 + lambda2 * fn  (paper Algorithm 1)
      obj1 = sum_{k != j} F_k(x)[c] - lambda1 * F_j(x)[c]   push target model j away from label c,
                                                            keep the others on c
      fn   = sum_k f_{n_k}(x)                               activate one uncovered neuron per model
    Returns the RMS-normalized gradient d(obj)/d(x).
    '''
    x = tf.convert_to_tensor(gen_img)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss_diff, loss_neuron = 0.0, 0.0
        for k, ((names, extractor), (layer_name, index)) in enumerate(zip(extractors, neurons)):
            outputs = dict(zip(names, extractor(x, training=False)))
            logit_c = tf.reduce_mean(outputs['before_softmax'][..., orig_label])
            loss_diff += -args.weight_diff * logit_c if k == args.target_model else logit_c
            loss_neuron += tf.reduce_mean(outputs[layer_name][..., index])
        final_loss = loss_diff + args.weight_nc * loss_neuron
    grads = tape.gradient(final_loss, x)
    return normalize(grads).numpy()


# ==============================================================================================
# Step 5: main loop over seeds
# ==============================================================================================
found = 0
records = []
for seed_idx in range(args.seeds):
    gen_img = np.expand_dims(random.choice(x_test), axis=0).copy()
    orig_img = gen_img.copy()
    record = {'seed_idx': seed_idx, 'status': 'failed', 'iterations': None, 'predictions': None, 'files': None}
    records.append(record)

    # 5a: if the models already disagree on the seed, it is a difference-inducing input as-is
    label1, label2, label3 = predict_labels(gen_img)
    record['orig_predictions'] = [label1, label2, label3]
    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2, label3)
              + bcolors.ENDC)
        record_and_report(gen_img)
        fname = '%03d_already_differ_%d_%d_%d.png' % (seed_idx, label1, label2, label3)
        imsave(os.path.join(args.out_dir, fname), deprocess_image(gen_img))
        record.update(status='already_differ', iterations=0, predictions=[label1, label2, label3], files=[fname])
        record['coverage'] = [neuron_covered(d)[2] for d in model_layer_dicts]
        found += 1
        continue

    # 5b: all models agree on label c; pick one uncovered neuron per model to activate
    orig_label = label1
    neurons = [neuron_to_cover(d) for d in model_layer_dicts]

    # 5c: gradient ascent on the joint objective, under the domain constraint
    for iters in range(args.grad_iterations):
        grads_value = joint_loss_and_grads(gen_img, orig_label, neurons)
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        gen_img += grads_value * args.step
        if not args.no_clip:
            # paper Sec. 4.2: generated pixels must stay in the valid range
            gen_img = np.clip(gen_img, 0, 1)

        # 5d: stop as soon as the models disagree; record coverage and save the pair of images
        predictions1, predictions2, predictions3 = predict_labels(gen_img)
        if not predictions1 == predictions2 == predictions3:
            print(bcolors.OKBLUE + 'seed %d (label %d) -> %d, %d, %d after %d iterations'
                  % (seed_idx, orig_label, predictions1, predictions2, predictions3, iters + 1) + bcolors.ENDC)
            record_and_report(gen_img)
            prefix = '%03d_%s_%d_%d_%d' % (seed_idx, args.transformation, predictions1, predictions2, predictions3)
            imsave(os.path.join(args.out_dir, prefix + '.png'), deprocess_image(gen_img))
            imsave(os.path.join(args.out_dir, prefix + '_orig.png'), deprocess_image(orig_img))
            record.update(status='generated', iterations=iters + 1,
                          predictions=[predictions1, predictions2, predictions3],
                          files=[prefix + '.png', prefix + '_orig.png'])
            found += 1
            break
    record['coverage'] = [neuron_covered(d)[2] for d in model_layer_dicts]

print(bcolors.HEADER + 'Done: %d / %d seeds produced difference-inducing inputs' % (found, args.seeds) + bcolors.ENDC)

# ==============================================================================================
# Step 6: log the run for visualize.py
# ==============================================================================================
with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
    json.dump({'args': vars(args),
               'neurons_per_model': [len(d) for d in model_layer_dicts],
               'records': records}, f, indent=1)
