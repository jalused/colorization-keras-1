import numpy as np

def rand_patches(x, y, opts):
    x_patch_size = opts["img_patch_size"]
    y_patch_size = opts["color_patch_size"]
    x_pixel_patch_size = opts["img_pixel_feature_patch_size"]
    num_patches= opts["num_patches"]
    
    img_col_size = x.shape[1]
    img_row_size = x.shape[2]
    img_num = x.shape[0]

    patch_x = np.zeros((num_patches, x_patch_size, x_patch_size))
    patch_x_pixel = np.zeros((num_patches, x_pixel_patch_size, x_pixel_patch_size))
    patch_y = np.zeros((num_patches, y.shape[1], y_patch_size * y_patch_size)) 

    for i in range(num_patches):
        img = np.random.randint(0, img_num, (1))
        row = np.random.randint((x_patch_size - 1) / 2, img_row_size - (x_patch_size - 1) / 2, (1));
        col = np.random.randint((x_patch_size - 1) / 2, img_col_size - (x_patch_size - 1) / 2, (1));
        patch_x[i, :, :] = x[img, col - (x_patch_size - 1) / 2 : col + (x_patch_size - 1) / 2 + 1, row - (x_patch_size - 1) / 2 : row + (x_patch_size - 1) / 2 + 1]
        patch_x_pixel[i, :, :] = x[img, col - (x_pixel_patch_size - 1) / 2 : col + (x_pixel_patch_size - 1) / 2 + 1, row - (x_pixel_patch_size - 1) / 2 : row + (x_pixel_patch_size - 1) / 2 + 1]
        selected_y = y[img, :, :].reshape(y.shape[1], x.shape[1], x.shape[2])
        #selected_y = selected_y.reshape(2, selected_y.size / 2)

        # print selected_y[0, col, row]
        # print selected_y[1, col, row]

        selected_y = selected_y[:, col - (y_patch_size - 1) / 2 : col + (y_patch_size - 1) / 2 + 1, row - (y_patch_size - 1) / 2 : row + (y_patch_size - 1) / 2 + 1]
        selected_y = selected_y.reshape(selected_y.shape[0], selected_y.shape[1] * selected_y.shape[2])
        patch_y[i, :, :] = selected_y
    return [patch_x, patch_x_pixel, patch_y]

def split_test_data(x, y, opts):
    im_num = x.shape[0]
    img_patch_size = opts["img_patch_size"]
    img_pixel_patch_size = opts["img_pixel_feature_patch_size"]
    color_patch_size = opts["color_patch_size"]
    img_row_size = x.shape[1]
    img_col_size = x.shape[2]
    patch_row_num = img_row_size - img_patch_size + 1
    patch_col_num = img_col_size - img_patch_size + 1
    patch_num = patch_row_num * patch_col_num
    x_patches = np.zeros((im_num, patch_num, img_patch_size, img_patch_size))
    x_pixel_patches = np.zeros((im_num, patch_num, img_pixel_patch_size, img_pixel_patch_size))
    y_patches = np.zeros((im_num, patch_num, y.shape[1], color_patch_size * color_patch_size))
    for i in range(im_num):
        for col in range((img_patch_size - 1) / 2, (img_patch_size - 1) / 2 + patch_col_num):
            for row in range((img_patch_size - 1) / 2,  (img_patch_size - 1) / 2 + patch_row_num):
                #print (col - (img_patch_size - 1) / 2) * patch_col_num + col - (img_patch_size) / 2
                img_patch = x[i, col - (img_patch_size - 1) / 2 : col + (img_patch_size + 1) / 2, row - (img_patch_size - 1) / 2 : row + (img_patch_size + 1) / 2]
                img_pixel_patch = x[i, col - (img_pixel_patch_size - 1) / 2 : col + (img_pixel_patch_size + 1) / 2, row - (img_pixel_patch_size - 1) / 2 : row + (img_pixel_patch_size + 1) / 2]

                #print img_patch.shape
                # if (img_patch.shape[0] != 7) or (img_patch.shape[1] != 7):
                #     print "wrong"
                #     print row
                #     print col
                x_patches[i, (col - (img_patch_size - 1) / 2) * patch_col_num + row - (img_patch_size - 1) / 2, :, :] = img_patch
                x_pixel_patches[i, (col - (img_patch_size - 1) / 2) * patch_col_num + row - (img_patch_size - 1) / 2, :, :] = img_pixel_patch
                temp_y = y[i, :, :].reshape(y.shape[1], x.shape[1], x.shape[2])
                color_patch = temp_y[:, col - (color_patch_size - 1) / 2 : col + (color_patch_size + 1) / 2, row - (color_patch_size - 1) / 2 : row + (color_patch_size + 1) / 2]
                color_patch = color_patch.reshape(y.shape[1], color_patch.shape[1] * color_patch.shape[2])
                #print row * patch_row_num + col
                y_patches[i, (col - (img_patch_size - 1) / 2) * patch_col_num + row - (img_patch_size - 1) / 2, :, :] = color_patch
    return [x_patches, x_pixel_patches, y_patches]

def rgb2yuv(r, g, b):
    rm = np.double(np.asmatrix(r)) / 255
    gm = np.double(np.asmatrix(g)) / 255
    bm = np.double(np.asmatrix(b)) / 255
    y = 0.299*r + 0.587*g + 0.114*b;
    u = -0.147*r - 0.289*g + 0.436*b;
    v = 0.615*r - 0.515*g - 0.100*b;
    return [y, u, v]

def yuv2rgb(y, u, v):
   r = y + 1.14 * v
   g = y - 0.39*u - 0.58*v
   b= y + 2.03*u
   
   r_label = r > 0
   r = r * r_label
   r_label = r < 1
   r = r ** r_label
   g_label = g > 0
   g = g * g_label
   g_label = g < 1
   g = g ** g_label
   b_label = b > 0
   b = b * b_label
   b_label = b < 1
   b = b ** b_label
   r = r * 255
   g = g * 255
   b = b * 255

   r = np.uint8(r)
   g = np.uint8(g)
   b = np.uint8(b)
   return [r, g, b]
