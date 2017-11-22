from __future__ import division, print_function, absolute_import

import os

import cv2
import numpy as np
import skimage
import skimage.io
import skimage.util

import config
import selectivesearch
import tools


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


# Clip Image
def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, save_path, threshold=0.6, is_svm=False, save=False):
    fr = open(datafile, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    for num, line in enumerate(train_list):
        labels = []
        images = []
        label0s = []
        image0s = []
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = skimage.io.imread(tmp[0])
        img = skimage.util.img_as_float(img)
        m = [np.mean(img[:,:,i]) for i in range(3)]
        std = [np.std(img[:,:,i]) for i in range(3)]
        img = [(img[:,:,i] - m[i]) / std[i] for i in range(3)]
        
        img = np.array(img)
        img = np.transpose(img, [1, 2, 0])
        img_lbl, regions = selectivesearch.selective_search(
                               img, scale=0.2, sigma=0.8, min_size=10)
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding small regions
            if r['size'] < 300:
                continue
            if (r['rect'][2] * r['rect'][3]) < 400:
                continue
            # resize to 227 * 227 for input
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # Delete Empty array
            if len(proposal_img) == 0:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            # Check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if is_svm:
                if iou_val < 0.3:
                    image0s.append(img_float)
                    label0s.append(0)
                else:
                    if iou_val > threshold:
                        labels.append(index)
                        images.append(img_float)
            else:
                label = np.zeros(num_clss + 1)
                if iou_val < 0.3:
                    label[0] = 1
                    image0s.append(img_float)
                    label0s.append(label)
                else:
                    if iou_val > threshold:
                        label[index] = 1
                        labels.append(label)
                        images.append(img_float)
        label_index = np.random.randint(len(label0s), size = len(labels) * 2)
        print('bg %d, obj %d, samp %d'%(len(label0s), len(labels), len(label_index)))
        for index in label_index:
            images.append(image0s[index])
            labels.append(label0s[index])
        tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
        if save:
            np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'), [images, labels])
    print(' ')
    fr.close()


# load data
def load_from_npy(data_set):
    images, labels = [], []
    data_list = os.listdir(data_set)
    # random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d))
        images.extend(i)
        labels.extend(l)
        tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(' ')
    return images, labels

def calcIOU(rect1, rect2):  
    one_x, one_y, one_w, one_h = rect1
    two_x, two_y, two_w, two_h = rect2
    if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):  
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))  
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))  
  
        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))  
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))  
  
        inter_w = abs(rd_x_inter - lu_x_inter)  
        inter_h = abs(lu_y_inter - rd_y_inter)  
  
        inter_square = inter_w * inter_h  
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square  
        calcIOU = inter_square / union_square * 1.0  
    else:  
        calcIOU = 0
    return calcIOU  