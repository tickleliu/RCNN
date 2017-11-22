from __future__ import division, print_function, absolute_import

import os
import os.path
import traceback

import cv2
import numpy as np
import tflearn
import skimage.io
import skimage.util
from sklearn import svm
from sklearn.externals import joblib
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import tensorflow as tf

import config
import preprocessing_RCNN as prep
import selectivesearch
import train_svm
import nms


def image_proposal(img_path):
    img = cv2.imread(img_path)
    img = skimage.io.imread(tmp[0])
    img = skimage.util.img_as_float(img)
    m = [np.mean(img[:, :, i]) for i in range(3)]
    std = [np.std(img[:, :, i]) for i in range(3)]
    img = [(img[:, :, i] - m[i]) / std[i] for i in range(3)]
    img = np.array(img)
    img = np.transpose(img, [1, 2, 0])
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


#############################################################
def eval_map(rin, sample_count, threshold=0.5):
    rin = rin[(-rin[:, 4]).argsort()]
    tp = 0
    fp = 0
    pr = []
    for item in rin:

        # if item[4] < 0.8:
        #     continue
        rect1 = item[0], item[1], item[2], item[3]
        rect2 = item[5], item[6], item[7], item[8]
        ov = prep.calcIOU(rect1, rect2)
        print("over lap: %f"%ov)
        if ov >= threshold:
            tp = tp + 1
            prt = tp / (tp + fp)
            pr.append(prt)
        else:
            fp = fp + 1
    print(pr)
    mean_ap = np.sum(pr) / sample_count
    return mean_ap


if __name__ == '__main__':
    net = create_alexnet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)

    train_file_folder = config.TRAIN_SVM
    svms = []
    svrxs = []
    svrys = []
    svrws = []
    svrhs = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
            svrxs.append(joblib.load(os.path.join(train_file_folder, file)))
            svrys.append(joblib.load(os.path.join(train_file_folder, file)))
            svrws.append(joblib.load(os.path.join(train_file_folder, file)))
            svrhs.append(joblib.load(os.path.join(train_file_folder, file)))

    # train_svr_file_folder = config.TRAIN_SVR
    # for file in os.listdir(train_svr_file_folder):
    #     if file.split('_')[-1] == 'svm@x.pkl':
    #         svrxs.append(joblib.load(os.path.join(train_svr_file_folder, file)))
    #
    #     if file.split('_')[-1] == 'svm@y.pkl':
    #         svrys.append(joblib.load(os.path.join(train_svr_file_folder, file)))
    #
    #     if file.split('_')[-1] == 'svm@w.pkl':
    #         svrws.append(joblib.load(os.path.join(train_svr_file_folder, file)))
    #
    #     if file.split('_')[-1] == 'svm@h.pkl':
    #         svrhs.append(joblib.load(os.path.join(train_svr_file_folder, file)))


    print("Done loading svms")

    ###############################################################################
    # evaluate
    fr = open(config.TEST_LIST, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    print("total test sample: %d" % (len(train_list)))
    map_results = []
    for num, line in enumerate(train_list):
        try:
            tmp = line.strip().split(' ')
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]

            img_path = tmp[0]
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
            results_dict = []

            count = 0

            for f in features:
                for svm, svrx, svry, svrw, svrh in zip(svms, svrxs, svrys, svrws, svrhs):
                    pred = svm.predict([f.tolist()])
                    # not background
                    prob = svm.predict_proba([f.tolist()])
                    # if pred[0] != 0:
                    if prob[0][1] > 0.4:
                        if verts[count][2] * verts[count][3] < width * height / 3:

                            verts[count] = list(verts[count])

                            # bouding box regression
                            # tx = svrx.predict([f.tolist()])
                            # ty = svry.predict([f.tolist()])
                            # tw = svrw.predict([f.tolist()])
                            # th = svrh.predict([f.tolist()])
                            # verts[count][0] = tx * verts[count][2] + verts[count][0]
                            # verts[count][1] = ty * verts[count][3] + verts[count][1]
                            # verts[count][2] = np.exp(tw) * verts[count][2]
                            # verts[count][3] = np.exp(th) * verts[count][3]

                            print(verts[count])
                            results.append(verts[count])

                            verts[count][2] = verts[count][0] + verts[count][2]
                            verts[count][3] = verts[count][1] + verts[count][3]
                            dict_item = {'index': count, 'rect': verts[count], 'prob': svm.predict_proba([f.tolist()])}
                            results_dict.append(dict_item)
                count += 1

            results = []
            for item in results_dict:
                temp = item['rect']
                temp.append(item['prob'][0][1])
                results.append(temp)

            print(results)

            results = np.array(results)
            print(results.shape)
            results_in = nms.nms(results, 0.3)
            # map_results.extend(results_in)
            print(results_in)
            results = results[results_in, :]
            # img_ori = img_path.split('/')[-1]
            # print('ori/' + img_ori)
            #             img = cv2.imread('ori/' + img_ori)
            # img = cv2.imread(img_path)
            # print(img.shape)

            for item in results:
                x, y, w, h = map(int, tmp[2].split(','))
                # item[0] = item[0].tolist()
                map_results.append(
                    [item[0], item[1], item[2] - item[0], item[3] - item[1], item[4], x, y, w, h])
                # print(map_results)
            #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)  # B,G,R
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.imwrite("results/%d.jpg" % num, img)

        except Exception as e:
            traceback.print_exc()
    print(map_results)
    print(np.array(map_results).shape)
    print("mean average precision: %f"%eval_map(np.array(map_results), len(train_list)))
