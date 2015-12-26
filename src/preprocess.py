#!/usr/bin/env python
from PIL import Image
import subprocess
import sys
import numpy as np
from util import rgb2yuv
from util import yuv2rgb
resized_image_path = '/home/curio/datasets/flickr8k/flicker_image/flicker8k_resized_image/';
yuv_image_path = '/home/curio/code/colorization-keras/data/yuv_image/';

try:
    output = subprocess.check_output('ls ' + resized_image_path, shell = True);
except subprocess.CalledProcessError:
    print 'ls error'
    sys.exit()
images = output.split('\n')
images = images[0 : len(images) - 1]
#print len(images)
im_row = 256
im_col = 256
im_num = len(images)
train_num = int(0.9 * len(images))
test_num = im_num - train_num

mini_train_num = 30
mini_test_num = 40

train_x = np.zeros((train_num, im_row, im_col))
train_y = np.zeros((train_num, 2, im_row * im_col))

dataset_path = "/home/jiangliang/code/colorization-keras/data/yuv_image/"
train_file = "train.npz"
mini_train_file = "mini_train.npz"

test_x = np.zeros((test_num, im_row, im_col))
test_y = np.zeros((test_num, 2, im_row * im_col))

for i in range(len(images)):
    print str(i) + "\t" + images[i]
    im = Image.open(resized_image_path + images[i])
    #im.show()
    array = np.array(im)

    print "array.shape" + str(array.shape)
    #print type(array)
    #array = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    #print type(array)
    #print array.shape

    r = array[:, :, 0]
    g = array[:, :, 1]
    b = array[:, :, 2]
    rm = np.double(np.asmatrix(r)) / 255
    gm = np.double(np.asmatrix(g)) / 255
    bm = np.double(np.asmatrix(b)) / 255
    [ym, um, vm] = rgb2yuv(rm ,gm, bm)
    # [rm1, gm1, bm1] = yuv2rgb(ym, um, vm)
    # rm1 = np.uint8(rm1 * 255)
    # gm1 = np.uint8(gm1 * 255)
    # bm1 = np.uint8(bm1 * 255)
    # ra = np.asarray(rm1)
    # ga = np.asarray(gm1)
    # ba = np.asarray(bm1)
    #
    # shape = ra.shape
    #
    # converted_rgb = np.zeros((shape[0], shape[1], 3), dtype="uint8")
    # #print "converted_rgb.shape: " + str(converted_rgb.shape)
    # converted_rgb[:, :, 0] = ra
    # converted_rgb[:, :, 1] = ga
    # converted_rgb[:, :, 2] = ba
    # #converted_rgb = np.array([[ra], [ga], [ba]])
    # max = np.max(converted_rgb)
    # min = np.min(converted_rgb)
    # #print "max: " + str(max) + ", min: " + str(min)
    #
    # converted_im = Image.fromarray(converted_rgb)
    # converted_im  = Image.fromarray(converted_rgb)
    # converted_im.show()

    shape = um.shape

    u = np.reshape(um, (shape[0] * shape[1]))
    print "u.shape: " + str(u.shape)
    shape = vm.shape
    v = np.reshape(vm, (shape[0] * shape[1]))

    if i < train_num:
        train_x[i, :, :] = ym;
        train_y[i, 0, :] = u;
        train_y[i, 1, :] = v;
    else:
        test_x[i - train_num, :, :] = ym;
        test_y[i - train_num, 0, :] = u;
        test_y[i - train_num, 1, :] = v;

    # train_x = train_x[0, :, :]
    # train_y = train_y[0, :, :]
    #
    # y = train_x.reshape(256, 256)
    # uv = train_y.reshape(2, 256, 256)
    # u = uv[0, :, :]
    # v = uv[1, :, :]
    #
    # [r, g, b] = yuv2rgb(y, u, v)
    # rm1 = np.uint8(r * 255)
    # gm1 = np.uint8(g * 255)
    # bm1 = np.uint8(b * 255)
    # ra = np.asarray(rm1)
    # ga = np.asarray(gm1)
    # ba = np.asarray(bm1)
    #
    # converted_rgb = np.zeros((256, 256, 3), dtype = 'uint8')
    # converted_rgb[:, :, 0] = ra
    # converted_rgb[:, :, 1] = ga
    # converted_rgb[:, :, 2] = ba
    #
    # im = Image.fromarray(converted_rgb)
    # im.show()


temp = np.arange(train_num)
#np.random.shuffle(temp)
mini_train = temp[0 : mini_train_num]
mini_train_x = train_x[mini_train, :, :]
mini_train_y = train_y[mini_train, :, :]

temp = np.arange(test_num)
#np.random.shuffle(temp)
mini_test = temp[0 : mini_test_num]
mini_test_x = test_x[mini_test, :, :]
mini_test_y = test_y[mini_test, :, :]
np.savez(dataset_path + train_file, train_x = train_x, test_x = test_x, train_y = train_y, test_y = test_y)
np.savez(dataset_path + mini_train_file, train_x = mini_train_x, train_y = mini_train_y, test_x = mini_test_x, test_y = mini_test_y)
    # print type(ym)
    # #print type(ym)
    # #print ym.shape
    # [rm1, gm1, bm1] = yuv2rgb(ym, um, vm)
    # rm1 = np.uint8(rm1 * 255)
    # gm1 = np.uint8(gm1 * 255)
    # bm1 = np.uint8(bm1 * 255)
    # ra = np.asarray(rm1)
    # ga = np.asarray(gm1)
    # ba = np.asarray(bm1)
    # print "ra.shape: " + str(ra.shape)
    # print "ga.shape: " + str(ga.shape)
    # print "ba.shape: " + str(ba.shape)
    # shape = ra.shape
    # print "shape: " + str(shape)
    # converted_rgb = np.zeros((shape[0], shape[1], 3), dtype="uint8")
    # #print "converted_rgb.shape: " + str(converted_rgb.shape)
    # converted_rgb[:, :, 0] = ra
    # converted_rgb[:, :, 1] = ga
    # converted_rgb[:, :, 2] = ba
    # #converted_rgb = np.array([[ra], [ga], [ba]])
    # max = np.max(converted_rgb)
    # min = np.min(converted_rgb)
    # #print "max: " + str(max) + ", min: " + str(min)
    # print "converted_rgb[0, 0, 0].type: " + str(type(converted_rgb[0, 0, 0]))
    # print "converted_rgb.shape:" + str((converted_rgb[0, 0, 0]).shape)
    # print "converted_rgb[0, 0, 0]: " + str(converted_rgb[0, 0, 0])
    # print "onverted_rgb.shape: " + str(type(converted_rgb))
    # print converted_rgb
    # print converted_rgb
    # converted_im = Image.fromarray(converted_rgb)
    # converted_im  = Image.fromarray(converted_rgb)
    # converted_im.show()
    # re_im = Image.fromarray([rm1, gm1, bm1])
    # temp = np.abs(temp)
    # max = np.max(temp)
    # min = np.min(temp)
    # print max, min
    # temp = gm - gm1
    # temp = np.abs(temp)
    # max = np.max(temp)
    # min = np.min(temp)
    # print max, min
    # temp = bm - bm1
    # temp = np.abs(temp)
    # max = np.max(temp)
    # min = np.min(temp)
    # print max, min
    # a = ([(1, 2), (3, 4)])
    # am = np.asmatrix(a)
    # b = ([(2, 2), (2, 2)])
    # dm = np.asmatrix(b)
    # c = 3 * am + dm
    #
    # print type(c)
    # print am
    # print dm
    # print c
