import numpy as np
from config import WIDTH, HEIGHT, CLASS_LABEL_LIST
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, Model


# define vgg19 model (only change fully-layer)
def fineTuningModel():
    input_tensor = Input(shape=(WIDTH, HEIGHT, 3))
    vgg = VGG19(include_top=False, input_tensor=input_tensor)

    fc = Sequential()
    fc.add(Flatten(input_shape=vgg.output_shape[1:]))
    fc.add(Dense(units=256, activation='relu'))
    fc.add(Dropout(rate=0.5))
    fc.add(Dense(CLASS_LABEL_LIST, activation='softmax'))

    return Model(input=vgg.input, output=fc(vgg.output))
