# coding:utf-8
import keras
import sys, os
import scipy
import scipy.misc
import numpy as np
from keras.models import model_from_json

import json

imsize = (32, 32)
testpic = "./testpic/"
keras_model = "./test.json"
keras_param = "./test.hdf5"


def load_image(path):
    img = scipy.misc.imread(path, mode="RGB")
    img = scipy.misc.imresize(img, imsize)
    img = img / 255.0
    return img

def get_file(dir_path):
    """
    testpicディレクトリの配下の画像を1つのリストにして返す
    ['244573113_thumb.jpg', 'car1.jpg', 'car2.jpg', 'car3.jpg', 'cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg', 'dog5.jpg', 'dog6.jpg', 'dog7.jpg']
    """
    filenames = os.listdir(dir_path)
    return filenames

if __name__ == "__main__":

    pic = get_file(testpic)
    
    model = model_from_json(open(keras_model).read())
    model.load_weights(keras_param)
    model.summary()

    for i in pic:
        print(i) # ファイル名の出力
        img = load_image(testpic + i)
        #vec = model.predict(np.array([img]), batch_size=1)
        prd = model.predict(np.array([img]))
        print(prd)
        prelabel = np.argmax(prd, axis=1)

        # 各画像ファイルに猫ならファイル名+0が、犬ならファイル名+1、乗り物ならファイル名+2のラベルが付いている
        if prelabel == 0:
            print(">>> 猫")
        elif prelabel == 1:
            print(">>> 犬")
        elif prelabel == 2:
            print(">>> 乗り物")

        print("#"*55)