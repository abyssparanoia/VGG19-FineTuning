from config import *
from keras.preprocessing.image import ImageDataGenerator


def makeGenerator():
    trainResource = ImageDataGenerator(
        rescale=1/255,
        zoom_range=0.2
        horizontal_flip=True
    )

    trainGenerator = trainResource.flow_from_directory(
        PATH_TO_TRAIN_DATA,
        target_size=(WIDTH, HEIGHT),
        classes=CLASS_LABEL_LIST,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    validateResource = ImageDataGenerator(
        rescale=1/255
    )

    validateGenerator = validateResource.flow_from_directory(
        PATH_TO_VALIDATE_DATA,
        target_size=(WIDTH, HEIGHT),
        classes=CLASS_LABEL_LIST,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    return [trainGenerator, testGenerator]
