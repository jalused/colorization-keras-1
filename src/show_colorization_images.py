#!/usr/bin/env python
import numpy as np
from PIL import Image
from util import yuv2rgb

result_path = "../result/"
result_file = "result.npz"

images = np.load(result_path + result_file)

original_images = images["original_images"]
o_y = original_images[0, 0, :, :]
o_u = original_images[0, 1, :, :]
o_v = original_images[0, 2, :, :]
result_images = images["result_images"]
r_y = result_images[0, 0, :, :]
r_u = result_images[0, 1, :, :]
r_v = result_images[0, 2, :, :]

for i in range(original_images.shape[0]):
    or_im = original_images[i, :, :, :]
    re_im = result_images[i, :, :, :]

    [or_y, or_u, or_v] = yuv2rgb(np.asmatrix(or_im[0, :, :]), np.asmatrix(or_im[1, :, :]), np.asmatrix(or_im[2, :, :]))
    [re_y, re_u, re_v] = yuv2rgb(np.asmatrix(re_im[0, :, :]), np.asmatrix(re_im[1, :, :]), np.asmatrix(re_im[2, :, :]))

    original_image = np.zeros((or_y.shape[0], or_y.shape[1], 3), dtype = "uint8")
    result_image = np.zeros((re_y.shape[0], re_y.shape[1], 3), dtype = "uint8")

    original_image[:, :, 0] = np.asarray(or_y)
    original_image[:, :, 1] = np.asarray(or_u)
    original_image[:, :, 2] = np.asarray(or_v)

    result_image[:, :, 0] = np.asarray(re_y)
    result_image[:, :, 1] = np.asarray(re_u)
    result_image[:, :, 2] = np.asarray(re_v)

    original_image = Image.fromarray(original_image)
    result_image = Image.fromarray(result_image)
    
    gray_image = original_image.convert("L")
    original_image.show()
    gray_image.show()
    result_image.show()
