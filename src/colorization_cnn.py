#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import sgd, SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2

from PIL import Image
from util import *

# np.random.seed(2015)

data_path = "../data/yuv_image/"
data_file = "mini_train.npz"
result_path = "../result/"
result_file = "result.npz"
param_path = "../params/"
param_file = "param"
model_path = "../model/"
model_file = "model"

print("Loading data")
data = np.load(data_path + data_file)
train_x = data["train_x"]
test_x = data["test_x"]
train_y = data["train_y"]
test_y = data["test_y"]

opts = {}
opts["img_patch_size"] = 45
opts["img_pixel_feature_patch_size"] = 3
opts["num_patches"] = 65536
opts["color_patch_size"] = 1
opts["batch_size"] = 128
opts["epoch"] = 3
opts["train_flag"] = True

#train_x = train_x[6, :, :].reshape(1, train_x.shape[1], train_x.shape[2])
#train_y = train_y[6, :, :].reshape(1, train_y.shape[1], train_y.shape[2])
#
#y = train_x.reshape(256, 256)
#uv = train_y.reshape(2, 256, 256)
#u = uv[0, :, :]
#v = uv[1, :, :]
#
#[r, g, b] = yuv2rgb(y, u, v)
#ra = np.asarray(r)
#ga = np.asarray(g)
#ba = np.asarray(b)
#
#converted_rgb = np.zeros((256, 256, 3), dtype = 'uint8')
#converted_rgb[:, :, 0] = ra
#converted_rgb[:, :, 1] = ga
#converted_rgb[:, :, 2] = ba
#
#im = Image.fromarray(converted_rgb)
#im.show()

pixel_model = Sequential()
pixel_model.add(Flatten(input_shape=(opts["img_pixel_feature_patch_size"] * opts["img_pixel_feature_patch_size"], )))

texture_model = Sequential()
texture_model.add(ZeroPadding2D(padding = (2, 2), input_shape = (1, opts["img_patch_size"], opts["img_patch_size"])))
texture_model.add(Convolution2D(3, 5, 5, border_mode = 'valid', activation = 'relu'))
texture_model.add(ZeroPadding2D(padding = (2, 2)))
texture_model.add(Convolution2D(48, 5, 5, border_mode = 'valid', activation = 'relu'))
texture_model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
texture_model.add(ZeroPadding2D(padding = (2, 2)))
texture_model.add(Convolution2D(64, 5, 5, border_mode = 'valid', activation = 'relu'))
texture_model.add(ZeroPadding2D(padding = (2, 2)))
texture_model.add(Convolution2D(64, 5, 5, border_mode = 'valid', activation = 'relu'))
texture_model.add(MaxPooling2D(pool_size = (3, 3), strides = ([2, 2])))
texture_model.add(ZeroPadding2D(padding = (2, 2)))
texture_model.add(Convolution2D(64, 5, 5, border_mode = 'valid', activation = 'relu'))
texture_model.add(Flatten())
texture_model.add(Dense(256, activation = 'relu'))
texture_model.add(Dropout(0.5))

model = Sequential()
model.add(Merge([pixel_model, texture_model], mode = "concat"))
model.add(Dense(8 * opts["color_patch_size" * opts["color_patch_size"]], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(2 * opts["color_patch_size"] * opts["color_patch_size"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
model.add(Activation('sigmoid'))

print("Compiling model")
sgd = SGD(lr = 10e-4, momentum = 0.9, decay = 10e-4)
rms = RMSprop()
#sgd = SGD()
model.compile(loss = 'mean_squared_error', optimizer = rms)
#texture_model.compile(loss = 'mean_squared_error', optimizer = sgd)
yaml_model = texture_model.to_yaml()
open(model_path + model_file, "w").write(yaml_model)
#deal with command line parameters
if (len(sys.argv) > 1):
    if (sys.argv[1] == "train"):
        opts["train_flag"] = True
    elif (sys.argv[1] == "test"):
        opts["train_flag"] = False
    else:
        print("Wrong parameter")
        sys.exit

if (opts["train_flag"]):
    print("Get random patches")
    [train_x_patches, train_x_pixel_patches, train_y_patches] = rand_patches(train_x, train_y, opts)
    # [train_x_patches, train_x_pixel_patches, train_y_patches] = split_test_data(train_x, train_y, opts)
    #
    # train_x_patches = train_x_patches.reshape(train_x_patches.shape[1], 1, train_x_patches.shape[2], train_x_patches.shape[3])
    # train_x_pixel_patches = train_x_pixel_patches.reshape(train_x_pixel_patches.shape[1], train_x_pixel_patches.shape[2], train_x_pixel_patches.shape[3])
    # train_y_patches = train_y_patches.reshape(train_y_patches.shape[1], train_y_patches.shape[2], train_y_patches.shape[3])

    train_y_patches[:, 0, :] = train_y_patches[:, 0, :] * 2.294
    train_y_patches[:, 1, :] = train_y_patches[:, 1, :] * 1.626
    train_y_patches[:, 0, :] = 0.5 * (train_y_patches[:, 0, :] + 1)
    train_y_patches[:, 1, :] = 0.5 * (train_y_patches[:, 1, :] + 1)
    
    train_x_patches = train_x_patches.reshape(train_x_patches.shape[0], 1, opts["img_patch_size"], opts["img_patch_size"])
    train_x_vector = train_x_pixel_patches.reshape(train_x_pixel_patches.shape[0], train_x_pixel_patches.size / train_x_pixel_patches.shape[0])
    train_y_vector = train_y_patches.reshape(train_y_patches.shape[0], train_y_patches.size / train_y_patches.shape[0])

    print("Fitting")
    
    model.fit([train_x_vector, train_x_patches], train_y_vector, batch_size=opts["batch_size"], nb_epoch=opts["epoch"], show_accuracy=True, verbose=1)
    model.save_weights(param_path + param_file, overwrite=True)
    
    #texture_model.fit([train_x_patches], train_y_vector, batch_size=opts["batch_size"], nb_epoch=opts["epoch"], verbose=1)
    #texture_model.save_weights(param_path + param_file, overwrite=True)
else:
    print("Load Weights")
    model.load_weights(param_path + param_file)
    #texture_model.load_weights(param_path + param_file)

test_x = test_x[0, :, :].reshape(1, train_x.shape[1], train_x.shape[2])
test_y = test_y[0, :, :].reshape(1, train_y.shape[1], train_y.shape[2])

print("Splitting test data")
[test_x_patches, test_x_pixel_patches, test_y_patches] = split_test_data(test_x, test_y, opts)
test_x_patches = test_x_patches.reshape(test_x_patches.shape[0], test_x_patches.shape[1], 1, opts["img_patch_size"], opts["img_patch_size"])
test_x_vector = test_x_pixel_patches.reshape(test_x_pixel_patches.shape[0], test_x_pixel_patches.shape[1], test_x_pixel_patches.shape[2] * test_x_pixel_patches.shape[3])
test_y_vector = test_y_patches.reshape((test_y_patches.shape[0], test_y_patches.shape[1], test_y_patches.size / test_y_patches.shape[0] / test_y_patches.shape[1]))
original_image = np.zeros((test_x_vector.shape[0], 3, np.sqrt(test_x_vector.shape[1]), np.sqrt(test_x_vector.shape[1])))
result_image  = np.zeros((test_x_vector.shape[0], 3, np.sqrt(test_x_vector.shape[1]), np.sqrt(test_x_vector.shape[1])))

print("Evaluating")
for i in range(test_x_vector.shape[0]):
    x_patch = test_x_patches[i, :, :, :]
    x_vector = test_x_vector[i, :, :]
    y_vector = test_y_vector[i, :, :]

    temp = y_vector.copy()
    temp[:, 0] = temp[:, 0] * 2.294
    temp[:, 1] = temp[:, 1] * 1.626
    temp[:, 0] = 0.5 * (temp[:, 0] + 1)
    temp[:, 1] = 0.5 * (temp[:, 1] + 1)

    #[score, acc] = model.evaluate([x_vector, x_patch], temp, show_accuracy=True, verbose = 1)
    #[score, acc] = texture_model.evaluate([x_patch], temp, show_accuracy=True, verbose = 1)
    #print("score: " + str(score) + ", acc: " + str(acc))
    predict_color = model.predict([x_vector, x_patch], verbose = 1)
    #predict_color = texture_model.predict([x_patch], verbose = 1)
    predict_color = predict_color.transpose().reshape(2, np.sqrt(predict_color.shape[0]), np.sqrt(predict_color.shape[0]))

    print(str(np.sqrt(x_vector.shape[0])))
    im_size = int(np.sqrt(x_vector.shape[0]))
    print("image size: " + str(im_size))
    original_yuv = np.zeros((3, im_size, im_size))
    result_yuv = np.zeros((3, im_size, im_size))
    print(str(original_yuv.shape))
    y = x_vector[:, (x_vector.shape[1] - 1) / 2].reshape(im_size, im_size)
    original_yuv[0, :, :] = y
    result_yuv[0, :, :] = y
    original_yuv[1 : 3, :, :] = y_vector.transpose().reshape(2, im_size, im_size)
    original_image[i, :, :, :] = original_yuv
    result_yuv[1 : 3, :, :] = predict_color
    result_yuv[1, :, :] = 2 * result_yuv[1, :, :] - 1
    result_yuv[2, :, :] = 2 * result_yuv[2, :, :] - 1
    result_yuv[1, :, :] = result_yuv[1, :, :] / 2.294
    result_yuv[2, :, :] = result_yuv[2, :, :] / 1.626
    result_image[i, :, :, :] = result_yuv

    for i in range(7):
        for j in range(7):
            print("origin_u: " + str(original_yuv[1, i * 25, j * 25]) + ", pred_u: " + str(result_yuv[1, i * 25, j * 25]))
            print("origin_v: " + str(original_yuv[2, i * 25, j * 25]) + ", pred_v: " + str(result_yuv[2, i * 25, j * 25]))


np.savez(result_path + result_file, original_images = original_image, result_images = result_image)
