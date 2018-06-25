from FineTuning import fineTuningModel as ftModel
from keras.optimizers import SGD
from generator import makeGenerator
from config import *
from os.path import join


def train():
    model = ftModel()

    for layer in model.layers[:18]:
        layer.trainable = False

    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    trainGen, testGen = makeGenerator()

    _ = model.fit_generator(
        trainGen,
        samples_per_epoch=DATA_FOR_TRAIN,
        nb_epoch=EPOCH_SIZE,
        validation_data=testGen,
        nb_val_samples=DATA_FOR_VALIDATE
    )

    model.save_weights(join(PATH_TO_OUTPUT, 'vgg19-FineTuning.h5'))
