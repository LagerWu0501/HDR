import numpy as np
import cv2
import torch
import glob
from numpy import linalg as la
import sys
import matplotlib.pyplot as plt
import random

np.set_printoptions(threshold=sys.maxsize)


def random_select_pixels(images, pixel_num = 0):
    x = images[0].shape[0] - 1
    y = images[0].shape[1] - 1
    pixels = [[random.randint(0, x), random.randint(0, y)] for i in range(pixel_num)]
    return pixels

def gray(image):
    new_image = []
    for i in range(image.shape[0]):
        images = []
        for j in range(image.shape[1]):
            images.append((54*image[i][j][2]+183*image[i]
                          [j][1]+19*image[i][j][0])/256)
        new_image.append(images)
    new_image = np.array(new_image, dtype=np.uint8)
    return new_image


def recover_without_ghost_removal(images, log_delta_t, pixel_num, image_num, selected_pixels, g_constraint, Lambda):
    ## parameter descriptions ##
    # images: multiple RGB images
    # log_delta_t: natural logrithm of the exposure time
    # pixel_num: pixel number we have choosen
    # image_num: number of exposure images
    # selected_pixels: the axises of pixels we have choosen
    # g_constraint: to fix the g_map we want to find i.e. g_map[g_constraint] = 0


    images_LAB = np.array(images, dtype='int16')
    z_max = 255
    z_min = 0
    z_mid = (z_max + z_min) / 2
    w_map = np.minimum(abs(images_LAB - z_max), abs(images_LAB - z_min)) 

    print("solving least square problem")
    A = np.zeros((pixel_num * image_num + 255, 256 + pixel_num, 3), dtype='float')
    b = np.zeros((pixel_num * image_num + 255, 3), dtype='float')

    for i in range(pixel_num):
        w = w_map[:, selected_pixels[i][0], selected_pixels[i][1]]
        for j in range(image_num):
            A[i * image_num + j, images[j][selected_pixels[i][0], selected_pixels[i][1], 0], 0] = 1 * w[j, 0]
            A[i * image_num + j, images[j][selected_pixels[i][0], selected_pixels[i][1], 1], 1] = 1 * w[j, 1]
            A[i * image_num + j, images[j][selected_pixels[i][0], selected_pixels[i][1], 2], 2] = 1 * w[j, 2]
        A[i * image_num:i * image_num + image_num, 256 + i, :] = -1 * w
        b[i * image_num:i * image_num + image_num, 0] = log_delta_t * w[:, 0]
        b[i * image_num:i * image_num + image_num, 1] = log_delta_t * w[:, 1]
        b[i * image_num:i * image_num + image_num, 2] = log_delta_t * w[:, 2]
    A[pixel_num * image_num, g_constraint, :] = 1

    for i in range(pixel_num * image_num + 1, A.shape[0]):
        w = min(abs(i - (pixel_num * image_num + 1) + 1 - 0), abs(i - (pixel_num * image_num + 1) + 1 - 255))
        A[i, i - (pixel_num * image_num + 1), :] = 1 * w * Lambda
        A[i, i - (pixel_num * image_num), :] = -2 * w * Lambda
        A[i, i - (pixel_num * image_num - 1), :] = 1 * w * Lambda

    del w

    print("psuedo inverse", 0)
    pinv_A_0 = la.pinv(A[:, :, 0])
    x_0 = np.matmul(pinv_A_0, b[:, 0])
    del pinv_A_0

    print("psuedo inverse", 1)
    pinv_A_1 = la.pinv(A[:, :, 1])
    x_1 = np.matmul(pinv_A_1, b[:, 1])
    del pinv_A_1

    print("psuedo inverse", 2)
    pinv_A_2 = la.pinv(A[:, :, 2])
    x_2 = np.matmul(pinv_A_2, b[:, 2])
    del pinv_A_2

    del A
    del b


    g_maps = [x_0[:256], x_1[:256], x_2[:256]]
    g_maps = np.array(g_maps, dtype = 'float')

    print("construct")

    sum_weight = [w_map[:, :, :, 0].sum(axis = 0), w_map[:, :, :, 1].sum(axis = 0), w_map[:, :, :, 2].sum(axis = 0)]
    final_img = [g_maps[0, images[:, :, :, 0]], g_maps[1, images[:, :, :, 1]], g_maps[2, images[:, :, :, 2]]]
    for i in range(3):
        for img in range(len(images)):
            final_img[i][img] = final_img[i][img] - log_delta_t[img]
            final_img[i][img] = final_img[i][img] * w_map[img, :, :, i]
        final_img[i] = final_img[i].sum(axis = 0)
        final_img[i] = final_img[i] / (sum_weight[i] + 10**(-32))
    
    final_img = cv2.merge([final_img[0], final_img[1], final_img[2]])

    del sum_weight

    return final_img

def photographic(image):
    min_value=np.min(image)
    image+=abs(min_value)
    
    N = image.shape[0] * image.shape[1]
    delta = 10**(-10)

    high_key = 0.18
    # print(image)
    avg_Lw = np.exp(1/N * sum(sum(np.log(image + delta))))
    Lm = high_key / avg_Lw * image

    Lwhite = 1
    Ld = Lm * (1 + Lm / Lwhite**2) / (Lm + 1)
    Ld = cv2.normalize(Ld, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return Ld

def gama_mapping(image):
    Ldr = cv2.pow(image/255., 1.0/2)
    Ldr = cv2.normalize(Ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return Ldr


paths = glob.glob("./Cropped_Colored_Images/*")

print("read images")
log_delta_t = []
for path in paths:
    log_delta_t.append(np.log(1 / int(((path.split("./Cropped_Colored_Images")[1]).split("IMG_")[1]).split(".jpg")[0])))

images = []
for path in paths:
    images.append(cv2.imread(path))
images = np.array(images)

pixel_num = 2000
selected_pixels = random_select_pixels(images, pixel_num = pixel_num)

print("recover")
final_img = recover_without_ghost_removal(images, log_delta_t, pixel_num, len(paths), selected_pixels=selected_pixels, g_constraint=128, Lambda = 100)

gamma = gama_mapping(final_img)
photo = photographic(final_img)

img_float32 = np.float32(final_img)
final_img = cv2.normalize(img_float32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
im_color = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
im_color = cv2.applyColorMap((im_color).astype(np.uint8), cv2.COLORMAP_JET)

cv2.imwrite("./gamma.jpg", gamma)
cv2.imwrite("./photographic.jpg", photo)
cv2.imwrite("./applycolor.jpg", im_color)
