from __future__ import division, print_function, absolute_import

import os
import os.path
import random

import numpy as np
import skimage.io
import skimage.util
import tensorflow as tf
from sklearn import svm
from sklearn.externals import joblib
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import config
import preprocessing_RCNN
import selectivesearch
import tools


# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, save_path, threshold=0.6, save=False):
    fr = open(datafile, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    for num, line in enumerate(train_list):
        labels = []
        images = []
        label0s = []
        image0s = []
        g_rects = []
        p_rects = []
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = skimage.io.imread(tmp[0])
        img = skimage.util.img_as_float(img)
        m = [np.mean(img[:, :, i]) for i in range(3)]
        std = [np.std(img[:, :, i]) for i in range(3)]
        img = [(img[:, :, i] - m[i]) / std[i] for i in range(3)]

        img = np.array(img)
        img = np.transpose(img, [1, 2, 0])
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=0.2, sigma=0.9, min_size=10)
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
            proposal_img, proposal_vertice = preprocessing_RCNN.clip_pic(img, r['rect'])
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
            resized_proposal_img = preprocessing_RCNN.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = preprocessing_RCNN.IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if iou_val < threshold:
                pass
            else:
                labels.append(index)
                images.append(img_float)
                g_rects.append(ref_rect_int)
                p_rects.append([proposal_vertice[0], proposal_vertice[1], proposal_vertice[4], proposal_vertice[5]])
                print(g_rects[-1])
                print(p_rects[-1])
        tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
        if save:
            np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'),
                    [images,  g_rects, p_rects])
    print(' ')
    fr.close()


# load data
def load_from_npy(data_set):
    images, g_rects, p_rects = [], [], []
    data_list = os.listdir(data_set)
    random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, g, p = np.load(os.path.join(data_set, d))
        images.extend(i)
        g_rects.extend(g)
        p_rects.extend(p)
        tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(' ')
    print(np.array(g_rects).shape)
    print(np.array(p_rects).shape)
    return images, g_rects, p_rects

# Load training images
def generate_single_svr_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        load_train_proposals(train_file, 2, save_path, threshold=0.3, save=True)
    print("restoring svm dataset")
    images, rect_gs, rect_ps = load_from_npy(save_path)

    return images, rect_gs, rect_ps


# Use a already trained alexnet with the last layer redesigned
def create_alexnet():
    # Building 'AlexNet'
    in_put = input_data(shape=[None, 227, 227, 3], dtype=tf.float32)
    conv1 = conv_2d(in_put, 96, 11, strides=4, activation='relu', padding='same')
    lrn = local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    pool1 = max_pool_2d(lrn, 3, strides=2, padding='valid')
    lrn_1, lrn_2 = tf.split(pool1, 2, 3)
    conv2_1 = conv_2d(lrn_1, 128, 5, activation='relu', padding="same")
    conv2_2 = conv_2d(lrn_2, 128, 5, activation='relu', padding="same")
    conv2 = tf.concat([conv2_1, conv2_2], 3)
    lru = local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    pool2 = max_pool_2d(lru, 3, strides=2, padding='valid')
    conv3 = conv_2d(pool2, 384, 3, activation='relu', padding="same")

    conv3_1, conv3_2 = tf.split(conv3, 2, 3)
    conv4_1 = conv_2d(conv3_1, 192, 3, activation='relu', padding="same")
    conv4_2 = conv_2d(conv3_2, 192, 3, activation='relu', padding="same")
    #     conv4 = tf.concat([conv4_1, conv4_2], 3)

    conv5_1 = conv_2d(conv4_1, 128, 3, activation='relu', padding="same")
    conv5_2 = conv_2d(conv4_2, 128, 3, activation='relu', padding="same")
    conv5 = tf.concat([conv5_1, conv5_2], 3)

    pool3 = max_pool_2d(conv5, 3, strides=2, padding='valid')
    #     lru = local_response_normalization(pool3)
    fc1 = fully_connected(pool3, 4096, activation='relu')
    dp1 = dropout(fc1, 0.5)
    fc2 = fully_connected(dp1, 4096, activation='relu')
    #     dp2 = dropout(fc2, 0.5)
    network = regression(fc2, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


# Construct cascade svms
def train_bbox_svrs(train_file_folder, model):
    files = os.listdir(train_file_folder)
    svrs = []
    for train_file in files:
        if train_file.split('.')[-1] == 'txt':
            X, P, G = generate_single_svr_train(os.path.join(train_file_folder, train_file))
            P = np.array(P)
            G = np.array(G)
            tx = (G[:, 0] - P[:, 0]) / P[:, 2]
            ty = (G[:, 1] - P[:, 1]) / P[:, 3]
            tw = np.log(G[:, 2] / P[:, 2])
            th = np.log(G[:, 3] / P[:, 3])
            train_features = []
            for ind, i in enumerate(X):
                # extract features
                feats = model.predict([i])
                train_features.append(feats[0])
                tools.view_bar("extract features of %s" % train_file, ind + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))
            # results = [tx, ty, tw, th]
            # results = np.transpose(results, [1, 0])
            # print(np.shape(results))
            # SVM training
            clr_x = svm.SVR(kernel='linear')
            clr_y = svm.SVR(kernel='linear')
            clr_w = svm.SVR(kernel='linear')
            clr_h = svm.SVR(kernel='linear')
            print("fit svm")
            clr_x.fit(train_features, tx)
            clr_y.fit(train_features, ty)
            clr_w.fit(train_features, tw)
            clr_h.fit(train_features, th)
            svrs.append([clr_x, clr_y, clr_w, clr_h])
            joblib.dump(clr_x, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm@x.pkl'))
            joblib.dump(clr_y, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm@y.pkl'))
            joblib.dump(clr_w, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm@w.pkl'))
            joblib.dump(clr_h, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm@h.pkl'))
    return svrs


if __name__ == "__main__":
    train_file_folder = config.TRAIN_SVR

    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)
    svms = []
    print("start")
    svms = train_bbox_svrs(train_file_folder, model)
    print("Done fitting svrs")