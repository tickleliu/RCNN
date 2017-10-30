from __future__ import division, print_function, absolute_import

import os
import os.path
import traceback

import cv2
import numpy as np
import tflearn
from sklearn import svm
from sklearn.externals import joblib
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import config
import preprocessing_RCNN
import preprocessing_RCNN as prep
import selectivesearch
import tools


def image_proposal(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=1, sigma=0.9, min_size=80)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 500:
            continue
        if (r['rect'][2] * r['rect'][3]) < 1000:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
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
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


# Load training images
def generate_single_svm_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        prep.load_train_proposals(train_file, 2, save_path, threshold=0.3, is_svm=True, save=True)
    print("restoring svm dataset")
    images, labels = prep.load_from_npy(save_path)

    return images, labels


# Use a already trained alexnet with the last layer redesigned
def create_alexnet():
    # Building 'AlexNet'
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


# Construct cascade svms
def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder)
    svms = []
    for train_file in files:
        if train_file.split('.')[-1] == 'txt':
            X, Y = generate_single_svm_train(os.path.join(train_file_folder, train_file))
            train_features = []
            for ind, i in enumerate(X):
                # extract features
                feats = model.predict([i])
                train_features.append(feats[0])
                tools.view_bar("extract features of %s" % train_file, ind + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))
            # SVM training
            clf = svm.LinearSVC(class_weight='balanced')
            print("fit svm")
            clf.fit(train_features, Y)
            svms.append(clf)
            joblib.dump(clf, os.path.join(train_file_folder, str(train_file.split('.')[0]) + '_svm.pkl'))
    return svms


if __name__ == '__main__':
    train_file_folder = config.TRAIN_SVM

    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)
    svms = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
    if len(svms) == 0:
        svms = train_svms(train_file_folder, model)
    print("Done fitting svms")

    # evaluate
    fr = open(config.FINE_TUNE_LIST, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    for num, line in enumerate(train_list):
        try:
            tmp = line.strip().split(' ')
            img_path = tmp[0];
            imgs_t, verts_t = image_proposal(img_path)
            img = cv2.imread(img_path)
            [width, height, channel] = img.shape
            verts = []
            imgs = []
            for i in range(len(verts_t)):
                if verts_t[i][2] * verts_t[i][3] < width * height / 3 and verts_t[i][2] < width / 2 and verts_t[i][
                    3] < height / 2:
                    verts.append(verts_t[i])
                    imgs.append(imgs_t[i])

            print(len(imgs))
            features = model.predict(imgs)
            print("predict image:")
            print(np.shape(features))
            results = []
            results_label = []
            results_pre = []
            count = 0

            for f in features:
                for svm in svms:
                    pred = svm.predict([f.tolist()])
                    # not background
                    if pred[0] != 0:
                        if verts[count][2] * verts[count][3] < width * height / 3:
                            results.append(verts[count])
                            results_label.append(pred[0])
                            results_pre.append(pred)
                count += 1

            result_iou = []
            for result in results:
                result_iou_t = 0
                for result2 in results:
                    if result == result2:
                        continue
                    else:
                        iou = preprocessing_RCNN.calcIOU(result, result2)
                        if iou > 0.5:
                            result_iou_t = result_iou_t + iou
                result_iou.append(result_iou_t)
            # index = result_iou.index(max(result_iou))
            # results = results[index: index + 1]
            result_i = sorted(range(len(result_iou)), key=result_iou.__getitem__, reverse=True)
            result_i = result_i[0: 5]
            print(result_iou)
            print(np.asarray(result_iou)[result_i])
            img_ori = img_path.split('/')[-1]
            print('ori/' + img_ori)
            img = cv2.imread('ori/' + img_ori)
            print(img.shape)

            for item in result_i:
                x, y, w, h = results[item]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # B,G,R
            x, y, w, h = map(int, tmp[2].split(','))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imwrite("results/%d.jpg" % num, img)
        except Exception as e:
            traceback.print_exc()
