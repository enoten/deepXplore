import os
import random
from collections import defaultdict

import h5py
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model
from keras.utils import to_categorical
from PIL import Image

from configs import bcolors

IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10


# ----------------------------------------------------------------------------------------------
# Data / model helpers
# ----------------------------------------------------------------------------------------------

def load_mnist(one_hot=False):
    """Load MNIST as float32 images in [0, 1] with shape (N, 28, 28, 1)."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, IMG_ROWS, IMG_COLS, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, IMG_ROWS, IMG_COLS, 1).astype('float32') / 255
    if one_hot:
        y_train = to_categorical(y_train, NB_CLASSES)
        y_test = to_categorical(y_test, NB_CLASSES)
    return (x_train, y_train), (x_test, y_test)


def load_legacy_h5_weights(model, path):
    """Load a Keras 2.x `save_weights` .h5 file by layer name (format: <layer>/<layer>/kernel:0)."""
    with h5py.File(path, 'r') as f:
        for layer in model.layers:
            if not layer.weights:
                continue
            group = f[layer.name][layer.name]
            layer.set_weights([group['kernel:0'][()], group['bias:0'][()]])


def load_model_weights(model, name):
    """Prefer weights retrained with this code (<name>.weights.h5), else the original repo's <name>.h5."""
    here = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(here, name + '.weights.h5')
    legacy_path = os.path.join(here, name + '.h5')
    if os.path.exists(new_path):
        model.load_weights(new_path)
    else:
        load_legacy_h5_weights(model, legacy_path)
    print(bcolors.OKBLUE + name + ' loaded' + bcolors.ENDC)


def train_model(model, name, batch_size=256, epochs=10):
    """Train like the original repo (Adadelta, lr=1.0 = Keras 2 default) and save <name>.weights.h5."""
    import keras
    (x_train, y_train), (x_test, y_test) = load_mnist(one_hot=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(learning_rate=1.0),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)
    here = os.path.dirname(os.path.abspath(__file__))
    model.save_weights(os.path.join(here, name + '.weights.h5'))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\nOverall Test score:', score[0])
    print('Overall Test accuracy:', score[1])


# ----------------------------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------------------------

def deprocess_image(x):
    """(1, 28, 28, 1) float image in [0, 1] -> (28, 28) uint8 image."""
    x = np.clip(x * 255, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])


def imsave(path, img):
    Image.fromarray(img).save(path)


def normalize(x):
    """Normalize a gradient tensor by its RMS (L2 norm / sqrt(n)) so step size is scale independent."""
    return x / (tf.sqrt(tf.reduce_mean(tf.square(x))) + 1e-5)


# ----------------------------------------------------------------------------------------------
# Domain-specific constraints (paper Section 6.2): turn a raw gradient into an allowed perturbation
# ----------------------------------------------------------------------------------------------

def constraint_occl(gradients, start_point, rect_shape):
    """Single rectangle occlusion: only pixels inside the rectangle may change."""
    new_grads = np.zeros_like(gradients)
    r0, c0 = start_point
    h, w = rect_shape
    new_grads[:, r0:r0 + h, c0:c0 + w] = gradients[:, r0:r0 + h, c0:c0 + w]
    return new_grads


def constraint_light(gradients):
    """Lighting: every pixel moves by the same amount (mean gradient) -> image gets brighter/darker."""
    return np.mean(gradients) * np.ones_like(gradients)


def constraint_black(gradients, rect_shape=(6, 6)):
    """Multiple tiny black rectangles: pick a random 6x6 patch; darken it only if the gradient says so."""
    start_point = (random.randint(0, gradients.shape[1] - rect_shape[0]),
                   random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
                  start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


# ----------------------------------------------------------------------------------------------
# Neuron coverage (paper Section 4.1)
# ----------------------------------------------------------------------------------------------

def coverage_layer_names(model):
    """Layers whose output channels count as 'neurons' (everything except input and flatten)."""
    return [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]


def build_layer_extractor(model):
    """One Model that returns the outputs of every coverage layer in a single forward pass."""
    names = coverage_layer_names(model)
    extractor = Model(inputs=model.input, outputs=[model.get_layer(n).output for n in names])
    return names, extractor


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    """Each (layer_name, channel_index) is a neuron; all start uncovered."""
    for name in coverage_layer_names(model):
        for index in range(model.get_layer(name).output.shape[-1]):
            model_layer_dict[(name, index)] = False


def neuron_to_cover(model_layer_dict):
    """Pick a random not-yet-covered neuron (or any neuron if coverage is already 100%)."""
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        return random.choice(not_covered)
    return random.choice(list(model_layer_dict.keys()))


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    """Min-max scale a whole layer's output to [0, 1] so one threshold works for every layer."""
    lo, hi = intermediate_layer_output.min(), intermediate_layer_output.max()
    if hi - lo == 0:
        return np.zeros_like(intermediate_layer_output)
    return (intermediate_layer_output - lo) / (hi - lo) * (rmax - rmin) + rmin


def update_coverage(input_data, layer_names, extractor, model_layer_dict, threshold=0):
    """Mark neurons whose (scaled, spatially averaged) activation on input_data exceeds threshold."""
    outputs = extractor(input_data, training=False)
    for name, out in zip(layer_names, outputs):
        scaled = scale(np.asarray(out)[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold:
                model_layer_dict[(name, num_neuron)] = True


def full_coverage(model_layer_dict):
    return all(model_layer_dict.values())
