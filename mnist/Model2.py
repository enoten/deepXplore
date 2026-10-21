'''
LeNet-4

usage: python Model2.py   - retrain the model (writes Model2.weights.h5)
'''

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model

from utils import IMG_ROWS, IMG_COLS, NB_CLASSES, load_model_weights, train_model


def Model2(input_tensor=None, train=False):
    kernel_size = (5, 5)
    if input_tensor is None:
        input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, 1))

    # block1
    x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(84, activation='relu', name='fc1')(x)
    x = Dense(NB_CLASSES, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        train_model(model, 'Model2')
    else:
        load_model_weights(model, 'Model2')
    return model


if __name__ == '__main__':
    Model2(train=True)
