# coding:utf-8
import numpy as np
import scipy.misc
import tensorflow as tf
import pickle as p # python2系はcPickle
from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.datasets import cifar10

import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt

in_shape = (32, 32, 3)
cls_num = 3
data_dir = "./cifar-10-batches-py/"

def build_model():
    model = Sequential()
    
    model.add(Convolution2D(16, 3, 3, border_mode="same", input_shape=in_shape)) # big 73, nomal 48
    model.add(Activation("relu")) # 1

    model.add(Convolution2D(16, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 2

    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 3

    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 4

    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 5

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 6

    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 7

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu")) # 8

    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode="same")) # Conv〜reluセットで1層
    model.add(Activation("relu")) # 9
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100)) # 全結合層 # 10
    model.add(Activation("relu"))
    
    # 分類問題における出力層の設計として、一般的にソフトマックス関数を使う。
    model.add(Dropout(0.5))
    model.add(Dense(cls_num)) # 11
    model.add(Activation('softmax')) 

    model.compile(loss="categorical_crossentropy", 
        metrics   = ["accuracy"], 
        optimizer = "adam"
    )
    return model

def unpickle(file):
    fo = open(file, 'rb')
    dict = p.load(fo)
    fo.close()
    return dict

def change3class(exp_val, obj_val):
    new_x = [] # 説明変数(X)入力
    new_y = [] # 目的変数(Y)出力

    for (x, y) in zip(exp_val, obj_val): 
    # 数が同じじゃないとできない -> つまり、XとYは同じ枚数
        
        if y == 3:
            new_x.append(x)
            new_y.append(0)
        elif y == 5:
            new_x.append(x)
            new_y.append(1)
        elif y == 1:
            new_x.append(x)
            new_y.append(2)
        else:
            pass

    new_x = np.array(new_x)
    return new_x, new_y

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # print(y_train[10:20]) # 0〜49999(5万)

    # 訓練画像, 訓練ラベル
    X_train, y_train = change3class(X_train, y_train)

    # テスト画像, テストラベル
    X_test, y_test = change3class(X_test, y_test)

    y_train = np_utils.to_categorical(y_train)
    y_test  = np_utils.to_categorical(y_test)

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    X_train /= 255.0
    X_test  /= 255.0

    model = build_model()
    # 学習
    model.fit(X_train, y_train, 
        nb_epoch=250, 
        batch_size=128, # 訓練データに対して128個のミニバッチで学習
        validation_data=(X_test, y_test)
    )

    json_string = model.to_json()
    open('test.json', 'w').write(json_string)
    model.save_weights('test.hdf5')
    
    # evaluate
    score = model.evaluate(X_train, y_train)
    print("test loss", score[0])
    print("test acc",  score[1])
