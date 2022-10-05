import sys
import cv2
import numpy as np
import torch
import glob
import os
from numpy import linalg as la
import matplotlib.pyplot as plt
import random


def turn_gray(paths):
    images = []
    for path in paths:
        # print("processing : " + path)
        ori_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(ori_img)
    return np.array(images)

def little_alignment(images, scale_degree, noise_epsilon, alignment_epsilon, start_points):
    # resize the image 
    print("resize")
    target_imgs = []
    scale_level = 2**scale_degree
    for target in images:
        height = int(target.shape[0] / scale_level)
        width = int(target.shape[1] / scale_level)
        dim = (width, height)
        target_imgs.append( cv2.resize(target, dim) )
    target_imgs = np.array(target_imgs, dtype="int16")

    # make noise map and MTB
    print("make MTB and noise map")
    tar_noise_maps = []
    for img_i in range(len(target_imgs)):
        target_thres = int(np.median(target_imgs[img_i]))

        tar_noise_maps.append( np.array( ((((abs(target_imgs[img_i] - target_thres) - noise_epsilon ) & 128) ^128) / 127), dtype='int16'))
        target_imgs[img_i] = np.array( ((((target_imgs[img_i] - target_thres) & 128)^128) / 127), dtype='int16')
    tar_noise_maps = np.array(tar_noise_maps, dtype='int16')

    # alignment
    print("alignment")
    alignment_set = [[0, 0, 0, 0]]

    height = target_imgs[0].shape[0]
    width = target_imgs[0].shape[1]
    for img_index in range(1, len(target_imgs)):
        max_sum = ~(target_imgs[0] ^ target_imgs[img_index]) & tar_noise_maps[0] & tar_noise_maps[img_index]
        # print(max_sum)
        max_sum = sum(sum(max_sum))
        lower = 0
        left = 0
        upper = 0
        right = 0

        # lower left upper right
        point = start_points[img_index]
        lwl = max(0, point[0] - alignment_epsilon)
        lwr = min(height - 1, point[0] + alignment_epsilon)

        ll = max(0, point[1] - alignment_epsilon)
        lr = min(width - 1, point[1] + alignment_epsilon)

        upl = max(1, height - 1 - point[2] - alignment_epsilon)
        upu = min(height - 1, height - 1 - point[2] + alignment_epsilon)

        rl = max(1, width - 1 - point[3] - alignment_epsilon)
        rr = min(width - 1, width - 1 - point[3] + alignment_epsilon) 


        if (point[0] >= 0 and point[1] >= 0): ## direction right and upper
            for x in range(lwl, lwr + 1):
                for y in range(ll, lr + 1):
                    temp_sum = ~(target_imgs[0][x:, y:] ^ target_imgs[img_index][:height-x, :width-y]) & tar_noise_maps[0][x:, y:] & tar_noise_maps[img_index][:height-x, :width-y]
                    temp_sum = sum(sum(temp_sum))
                    if (temp_sum > max_sum):
                        max_sum = temp_sum
                        lower = x
                        left = y
                        upper = -1 * x
                        right = -1 * y
        if point[1] >= 0 and point[2] >= 0: ## direction right and lower
            for x in range(upl, upu + 1):
                for y in range(ll, lr + 1):
                    temp_sum = ~(target_imgs[0][:x, y:] ^ target_imgs[img_index][height-x:, :width-y]) & tar_noise_maps[0][:x, y:] & tar_noise_maps[img_index][height-x:, :width-y]
                    temp_sum = sum(sum(temp_sum))
                    if (temp_sum > max_sum):
                        max_sum = temp_sum
                        lower = -1 * (height - 1 - x)
                        left = y
                        upper = height - 1 - x
                        right = -1 * y
        if point[2] >= 0 and point[3] >= 0: ## direction left and lower
            for x in range(upl, upu + 1):
                for y in range(rl, rr + 1):
                    temp_sum = ~(target_imgs[0][:x, :y] ^ target_imgs[img_index][height-x:, width-y:]) & tar_noise_maps[0][:x, :y] & tar_noise_maps[img_index][height-x:, width-y:]
                    temp_sum = sum(sum(temp_sum))
                    if (temp_sum > max_sum):
                        max_sum = temp_sum
                        lower = -1 * (height - 1 - x)
                        left = -1 * (width  - 1 - y)
                        upper = height - 1 - x
                        right = width  - 1 - y
        if point[3] >= 0 and point[0] >= 0: ## direction left and upper
            for x in range(lwl, lwr + 1):
                for y in range(rl, rr + 1):
                    temp_sum = ~(target_imgs[0][x:, :y] ^ target_imgs[img_index][:height-x, width-y:]) & tar_noise_maps[0][x:, :y] & tar_noise_maps[img_index][:height-x, width-y:]
                    temp_sum = sum(sum(temp_sum))
                    if (temp_sum > max_sum):
                        max_sum = temp_sum
                        lower = x
                        left = -1 * width - 1 - y
                        upper = -1 * x
                        right = width - 1 - y
        
        
        alignment_set.append([lower, left, upper, right])
    return alignment_set

def alignment(images, max_scale_degree = 7, noise_epsilon = 10, alignment_epsilon = 10):
    # set start points
    print("set start points")
    start_points = []
    for i in range(len(images)):
        start_points.append([0, 0, 0, 0])
    
    # scale and alignment
    print("scale and alignment")
    for i in range(max_scale_degree + 1):
        scale_degree = max_scale_degree - i
        print(" scale", scale_degree)
        for x in range(len(start_points)):
            for y in range(len(start_points[0])):
                start_points[x][y] = 2 * start_points[x][y]
        start_points = little_alignment(images=images, scale_degree=scale_degree, noise_epsilon=noise_epsilon, alignment_epsilon=alignment_epsilon, start_points=start_points)

    for start_point in start_points:
        print(start_point)
    
    return start_points

    

def crop(paths, start_points, destination):
    offset = start_points
    for i in range(offset.shape[0]):
        for j in range(offset.shape[1]):
            if (offset[i][j] < 0):
                offset[i][j] = 0
    lower = max(offset[:, 0])
    left = max(offset[:, 1])
    upper = max(offset[:, 2])
    right = max(offset[:, 3])
    print(lower, left, upper, right)

    for img_i in range(len(paths)):

        ori_img = cv2.imread(paths[img_i])

        height = ori_img.shape[0]
        width = ori_img.shape[1]

        cropped_Img = ori_img[lower:height - upper, left:width - right]

        cv2.imwrite(destination + '/' + paths[img_i].split("/")[-1], cropped_Img)


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


def recover_without_ghost_removal(images, log_delta_t, image_num, pixel_num, selected_pixels, g_constraint=128, Lambda=100):

    z_max = 255
    z_min = 0
    z_mid = (z_max + z_min) / 2
    w_map = np.minimum(abs(images - z_max), abs(images - z_min)) 

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


def recover(images, log_delta_t, image_num, pixel_num, selected_pixels, g_constraint = 128, Lambda=100, max_iter=1):

    w_map = np.array(1 - (2*images/255 - 1)**12, dtype='float')
    # z_max = 255
    # z_min = 0
    # z_mid = (z_max + z_min) / 2
    # w_map = np.minimum(abs(images - z_max), abs(images - z_min)) 

    print("construct weight map")
    dim = 5 
    H = np.identity(dim)
    H_d = np.linalg.det(H)
    H_inv = np.linalg.inv(H)

    for t in range(max_iter):
        print(" iter", t)
        for r in range(images.shape[0]):
            print("  image", r)

            w_average = np.average(w_map[r], axis=2)  
            sum_w_pqs = np.zeros(w_map[r].shape)

            P_map = np.zeros((images[0].shape[0], images[0].shape[1], images[0].shape[2]))
            vector_space = np.zeros((images[0].shape[0], images[0].shape[1], dim))
            vector_space[:, :, :3] = images[r]

            for x in range(images[0].shape[0]):
                for y in range(images[0].shape[1]):
                    vector_space[x, y, 3] = x  
                    vector_space[x, y, 4] = y  

            # up and down
            sum_w_pqs[:-1, :, 0] += w_average[1:, :]
            sum_w_pqs[:-1, :, 1] += w_average[1:, :]
            sum_w_pqs[:-1, :, 2] += w_average[1:, :]
            sum_w_pqs[1:, :, 0] += w_average[:-1, :]
            sum_w_pqs[1:, :, 1] += w_average[:-1, :]
            sum_w_pqs[1:, :, 2] += w_average[:-1, :]
            # left and right
            sum_w_pqs[:, 1:, 0] = sum_w_pqs[:, 1:, 0] + w_average[:, :-1]
            sum_w_pqs[:, 1:, 1] = sum_w_pqs[:, 1:, 1] + w_average[:, :-1]
            sum_w_pqs[:, 1:, 2] = sum_w_pqs[:, 1:, 2] + w_average[:, :-1]
            sum_w_pqs[:, :-1, 0] = sum_w_pqs[:, :-1, 0] + w_average[:, 1:]
            sum_w_pqs[:, :-1, 1] = sum_w_pqs[:, :-1, 1] + w_average[:, 1:]
            sum_w_pqs[:, :-1, 2] = sum_w_pqs[:, :-1, 2] + w_average[:, 1:]
            # upper left and lower right
            sum_w_pqs[:-1, :-1, 0] = sum_w_pqs[:-1, :-1, 0] + w_average[1:, 1:]
            sum_w_pqs[:-1, :-1, 1] = sum_w_pqs[:-1, :-1, 1] + w_average[1:, 1:]
            sum_w_pqs[:-1, :-1, 2] = sum_w_pqs[:-1, :-1, 2] + w_average[1:, 1:]
            sum_w_pqs[1:, 1:, 0] = sum_w_pqs[1:, 1:, 0] + w_average[:-1, :-1]
            sum_w_pqs[1:, 1:, 1] = sum_w_pqs[1:, 1:, 1] + w_average[:-1, :-1]
            sum_w_pqs[1:, 1:, 2] = sum_w_pqs[1:, 1:, 2] + w_average[:-1, :-1]
            # upper right and lower left
            sum_w_pqs[1:, :-1, 0] = sum_w_pqs[1:, :-1, 0] + w_average[:-1, 1:]
            sum_w_pqs[1:, :-1, 1] = sum_w_pqs[1:, :-1, 1] + w_average[:-1, 1:]
            sum_w_pqs[1:, :-1, 2] = sum_w_pqs[1:, :-1, 2] + w_average[:-1, 1:]
            sum_w_pqs[:-1, 1:, 0] = sum_w_pqs[:-1, 1:, 0] + w_average[1:, :-1]
            sum_w_pqs[:-1, 1:, 1] = sum_w_pqs[:-1, 1:, 1] + w_average[1:, :-1]
            sum_w_pqs[:-1, 1:, 2] = sum_w_pqs[:-1, 1:, 2] + w_average[1:, :-1]


            # up and down
            vector = ((vector_space[:-1, :] - vector_space[1:, :])*(vector_space[:-1, :] - vector_space[1:, :])).sum(axis=2)
            P_map[:-1, :, 0] += w_average[1:, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, :, 1] += w_average[1:, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, :, 2] += w_average[1:, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, :, 0] += w_average[:-1, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, :, 1] += w_average[:-1, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, :, 2] += w_average[:-1, :] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            # left and right
            vector = ((vector_space[:, 1:] - vector_space[:, :-1])*(vector_space[:, 1:] - vector_space[:, :-1])).sum(axis=2)
            P_map[:, 1:, 0] += w_average[:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:, 1:, 1] += w_average[:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:, 1:, 2] += + w_average[:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:, :-1, 0] += w_average[:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:, :-1, 1] += w_average[:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:, :-1, 2] += w_average[:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            # upper left and lower right
            vector = ((vector_space[:-1, :-1] - vector_space[1:, 1:]) * (vector_space[:-1, :-1] - vector_space[1:, 1:])).sum(axis=2)
            P_map[:-1, :-1, 0] += w_average[1:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, :-1, 1] += w_average[1:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, :-1, 2] += w_average[1:, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, 1:, 0] += w_average[:-1, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, 1:, 1] += w_average[:-1, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, 1:, 2] += w_average[:-1, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            # upper right and lower left
            vector = ((vector_space[1:, :-1] - vector_space[:-1, 1:]) * (vector_space[1:, :-1] - vector_space[:-1, 1:])).sum(axis=2)
            P_map[1:, :-1, 0] += w_average[:-1, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, :-1, 1] += w_average[:-1, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[1:, :-1, 2] += w_average[:-1, 1:] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, 1:, 0] += w_average[1:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, 1:, 1] += w_average[1:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)
            P_map[:-1, 1:, 2] += w_average[1:, :-1] * H_d * (2 * np.pi)**(-dim / 2) * np.exp(-1/2 * vector)

            P_map[:, :] /= (sum_w_pqs[:, :] + 10**(-32))
            w_map[r] *= P_map

            norm = np.linalg.norm(w_map[r])
            w_map[r] /= norm
            w_map[r] *= 128

        del P_map
        del sum_w_pqs
        del vector_space
    # w_map = np.array(w_map, dtype='int16')

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

def photographic(image, high_key = 0.18, Lwhite = 1):
    min_value=np.min(image)
    image+=abs(min_value)
    
    N = image.shape[0] * image.shape[1]
    delta = 10**(-10)

    avg_Lw = np.exp(1/N * sum(sum(np.log(image + delta))))
    Lm = high_key / avg_Lw * image

    Ld = Lm * (1 + Lm / Lwhite**2) / (Lm + 1)
    Ld = cv2.normalize(Ld, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return Ld

def gamma_mapping(image):
    Ldr = cv2.pow(image/255., 1.0/2)
    Ldr = cv2.normalize(Ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return Ldr
    
def main(parameter):
    if len(parameter) == 1:
        images = glob.glob("Images/*")
        gray_images = turn_gray(images)
        start_points = np.array(alignment(gray_images), dtype='int16')

        destination = "Cropped_Colored_Images"
        crop(images, start_points, destination)

        paths = glob.glob("./Cropped_Colored_Images/*")
        log_delta_t = []
        for path in paths:
            log_delta_t.append(np.log(1 / int(((path.split("./Cropped_Colored_Images")[1]).split("IMG_")[1]).split(".jpg")[0])))

        images = []
        for path in paths:
            images.append(cv2.imread(path))
        images = np.array(images)

        pixel_num = 2000
        selected_pixels = random_select_pixels(images, pixel_num = pixel_num)

        final_img = recover_without_ghost_removal(images, log_delta_t, len(paths), pixel_num, selected_pixels=selected_pixels)

        gamma = gamma_mapping(final_img)
        photo = photographic(final_img)

        img_float32 = np.float32(final_img)
        final_img = cv2.normalize(img_float32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        im_color = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        im_color = cv2.applyColorMap((im_color).astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite("./gamma.jpg", gamma)
        cv2.imwrite("./photographic.jpg", photo)
        cv2.imwrite("./applycolor.jpg", im_color)



    elif parameter[1] == "-a":
        if (len(parameter) == 2):
            images = glob.glob("Images/*")
            gray_images = turn_gray(images)
            start_points = np.array(alignment(gray_images), dtype='int16')

            destination = "Cropped_Colored_Images"
            crop(images, start_points, destination)
        elif(len(parameter) == 7):
            images = glob.glob(parameter[2]+"/*")
            gray_images = turn_gray(images)
            
            max_scale_degree = int(parameter[3])
            noise_epsilon = int(parameter[4])
            alignment_epsilon = int(parameter[5])
            start_points = np.array(alignment(gray_images, max_scale_degree = max_scale_degree, noise_epsilon = noise_epsilon, alignment_epsilon = alignment_epsilon), dtype='int16')

            destination = parameter[6]
            crop(images, start_points, destination)
        else:
            print("parameter ERROR.")
            print("please check your parameters' format.")
            return

    elif parameter[1] == "-r":
        if (len(parameter) == 2):
            paths = glob.glob("./Cropped_Colored_Images/*")

            log_delta_t = []
            for path in paths:
                log_delta_t.append(np.log(1 / int(((path.split("./Cropped_Colored_Images")[1]).split("IMG_")[1]).split(".jpg")[0])))

            images = []
            for path in paths:
                images.append(cv2.imread(path))
            images = np.array(images, dtype='int16')

            pixel_num = 2000
            selected_pixels = random_select_pixels(images, pixel_num = pixel_num)

            final_img = recover_without_ghost_removal(images, log_delta_t, len(paths), pixel_num, selected_pixels=selected_pixels)

            gamma = gamma_mapping(final_img)
            photo = photographic(final_img)

            img_float32 = np.float32(final_img)
            final_img = cv2.normalize(img_float32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            im_color = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            im_color = cv2.applyColorMap((im_color).astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite("./gamma.jpg", gamma)
            cv2.imwrite("./photographic.jpg", photo)
            cv2.imwrite("./applycolor.jpg", im_color)

        elif (len(parameter) > 2 and parameter[2] == "-ng"):
            if (len(parameter) == 9):
                paths = glob.glob(parameter[3]+"/*")

                log_delta_t = []
                for path in paths:
                    log_delta_t.append(np.log(1 / int((path.split("IMG_")[1]).split(".jpg")[0])))

                images = []
                for path in paths:
                    images.append(cv2.imread(path))
                images = np.array(images, dtype='int16')

                pixel_num = int(parameter[4])
                selected_pixels = random_select_pixels(images, pixel_num = pixel_num)
                g_constraint = int(parameter[5])
                Lambda = int(parameter[6])

                final_img = recover_without_ghost_removal(images, log_delta_t, len(paths), pixel_num, selected_pixels=selected_pixels, g_constraint=g_constraint, Lambda = Lambda)

                high_key = float(parameter[7])
                L_white = float(parameter[8])
                gamma = gamma_mapping(final_img)
                photo = photographic(final_img, high_key, L_white)


                img_float32 = np.float32(final_img)
                final_img = cv2.normalize(img_float32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                im_color = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
                im_color = cv2.applyColorMap((im_color).astype(np.uint8), cv2.COLORMAP_JET)

                cv2.imwrite("./gamma.jpg", gamma)
                cv2.imwrite("./photographic.jpg", photo)
                cv2.imwrite("./radience.jpg")

            else:
                print("parameter ERROR.")
                print("please check your parameters' format.")
                return

        elif (len(parameter) > 2 and parameter[2] == "-wg"):
            if (len(parameter) == 10):
                paths = glob.glob(parameter[3]+"/*")

                log_delta_t = []
                for path in paths:
                    log_delta_t.append(np.log(1 / int((path.split("IMG_")[1]).split(".jpg")[0])))

                images = []
                for path in paths:
                    images.append(cv2.imread(path))
                images = np.array(images, dtype='int16')

                pixel_num = int(parameter[4])
                selected_pixels = random_select_pixels(images, pixel_num = pixel_num)

                g_constraint = int(parameter[5])
                Lambda = int(parameter[6])
                max_iter = int(parameter[7])

                final_img = recover(images, log_delta_t, len(paths), pixel_num, selected_pixels=selected_pixels, g_constraint=g_constraint, Lambda = Lambda, max_iter = max_iter)

                high_key = float(parameter[8])
                L_white = float(parameter[9])
                gamma = gamma_mapping(final_img)
                photo = photographic(final_img, high_key, L_white)

                img_float32 = np.float32(final_img)
                final_img = cv2.normalize(img_float32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                im_color = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
                im_color = cv2.applyColorMap((im_color).astype(np.uint8), cv2.COLORMAP_JET)

                cv2.imwrite("./gamma_gremoval.jpg", gamma)
                cv2.imwrite("./photographic_gremoval.jpg", photo)
                cv2.imwrite("./radience_gremoval.jpg", im_color)
            else:
                print("parameter ERROR.")
                print("please check your parameters' format.")
                return
        else:
            print("parameter ERROR.")
            print("please check your parameters' format.")
            return






if __name__=="__main__":
    main(sys.argv)