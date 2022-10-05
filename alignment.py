import cv2
import numpy as np
import torch
import glob
import os

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

def alignment(images, max_scale_degree, noise_epsilon, alignment_epsilon):
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

    
noise_epsilon = 10
alignment_epsilon = 10
max_scale_degree = 7

images = glob.glob("Images/*")
destination = "Cropped_Colored_Images"

gray_images = turn_gray(images, None)
start_points = np.array(alignment(gray_images, max_scale_degree = max_scale_degree, noise_epsilon = noise_epsilon, alignment_epsilon = alignment_epsilon), dtype='int16')
crop(images, start_points, destination)
