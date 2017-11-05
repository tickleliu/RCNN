from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
import os.path
import codecs
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
import config

from caffe_classes import class_names

net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

# Building 'AlexNet'
def create_alexnet(num_classes):
    in_put= input_data(shape=[None, 227, 227, 3], dtype=tf.float32)
    conv1 = conv_2d(in_put, 96, 11, strides=4, activation='relu', padding='same')
    lrn = local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    pool1 = max_pool_2d(lrn, 3, strides=2, padding='valid')
    lrn_1, lrn_2 = tf.split(pool1, 2, 3)
    conv2_1 = conv_2d(lrn_1, 128, 5, activation='relu', padding="same")
    conv2_2 = conv_2d(lrn_2, 128, 5, activation='relu', padding="same")
    conv2 = tf.concat([conv2_1, conv2_2], 3)
    lru = local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    pool2= max_pool_2d(lru, 3, strides=2, padding='valid')
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
#     dp1 = dropout(fc1, 0.5)
    fc2 = fully_connected(fc1, 4096, activation='relu')
#     dp2 = dropout(fc2, 0.5)
    fc3 = fully_connected(fc2, num_classes, activation='softmax')
#     network = fc3
    network = regression(fc3, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)
    
    model = tflearn.DNN(network)
    
    model.set_weights(fc3.W, net_data["fc8"][0])
    model.set_weights(fc3.b, net_data["fc8"][1])
    model.set_weights(fc2.W, net_data["fc7"][0])
    model.set_weights(fc2.b, net_data["fc7"][1])
    model.set_weights(fc1.W, net_data["fc6"][0])
    model.set_weights(fc1.b, net_data["fc6"][1])
    
    model.set_weights(conv5_1.W, np.split(np.array(net_data["conv5"][0]), 2, 3)[0])
    model.set_weights(conv5_1.b, np.split(np.array(net_data["conv5"][1]), 2, 0)[0])
    model.set_weights(conv5_2.W, np.split(np.array(net_data["conv5"][0]), 2, 3)[1])
    model.set_weights(conv5_2.b, np.split(np.array(net_data["conv5"][1]), 2, 0)[1])
    
    model.set_weights(conv4_1.W, np.split(np.array(net_data["conv4"][0]), 2, 3)[0])
    model.set_weights(conv4_1.b, np.split(np.array(net_data["conv4"][1]), 2, 0)[0])
    model.set_weights(conv4_2.W, np.split(np.array(net_data["conv4"][0]), 2, 3)[1])
    model.set_weights(conv4_2.b, np.split(np.array(net_data["conv4"][1]), 2, 0)[1])
    
    model.set_weights(conv3.W, net_data["conv3"][0])
    model.set_weights(conv3.b, net_data["conv3"][1])
    
    
    model.set_weights(conv2_1.W, np.split(np.array(net_data["conv2"][0]), 2, 3)[0])
    model.set_weights(conv2_1.b, np.split(np.array(net_data["conv2"][1]), 2, 0)[0])
    model.set_weights(conv2_2.W, np.split(np.array(net_data["conv2"][0]), 2, 3)[1])
    model.set_weights(conv2_2.b, np.split(np.array(net_data["conv2"][1]), 2, 0)[1])
    
    model.set_weights(conv1.W, net_data["conv1"][0])
    model.set_weights(conv1.b, net_data["conv1"][1])
    model.save(config.SAVE_MODEL_PATH)
    
    return model

if __name__ == '__main__':
    
    
    im1 = (imread("laska.png")[:,:,:3]).astype(np.float32)
    im1 = im1 - np.mean(im1)
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
    
    im2 = (imread("poodle.png")[:,:,:3]).astype(np.float32)
    im2 = im2 - np.mean(im2)
    im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]
    
    im3 = (imread("dog.png")[:,:,:3]).astype(np.float32)
    im3 = im3 - np.mean(im3)
    im3[:, :, 0], im3[:, :, 2] = im3[:, :, 2], im3[:, :, 0]
    
    im4 = (imread("dog2.png")[:,:,:3]).astype(np.float32)
    im4 = im4 - np.mean(im4)
    im4[:, :, 0], im4[:, :, 2] = im4[:, :, 2], im4[:, :, 0]
    
    model = create_alexnet(1000)
    output = model.predict(np.asarray([im1, im2, im3, im4]))
    print(output.shape)
    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind,:]
        print("Image", input_im_ind)
        for i in range(5):
            print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])