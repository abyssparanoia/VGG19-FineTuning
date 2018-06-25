from FineTuning import fineTuningModel as ftModel
from config import *
from os import listdir
from os.path import join
from keras.preprocessing import image
from numpy import expand_dims


def classify():
    model = ftModel()

    for imgFile in listdir(PATH_TO_TEST_DATA):
        print('now target image is ...', imgFile)
        img = image.load_img(
            path=join(PATH_TO_TEST_DATA, imgFile), target_size=(WIDTH, HEIGHT))
        imgData = image.img_to_array(img)
        x = expand_dims(imgData, axis=0) / 255
        pred = model.predict(x)[0]
        result = [(CLASS_LABEL_LIST[i], pred[i])
                  for i in pred.argsort()[-1:][::-1]]
        print('classify result is ', result, '\n')
