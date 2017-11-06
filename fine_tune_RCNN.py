from __future__ import division, print_function, absolute_import
import os.path
import preprocessing_RCNN as prep
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf


# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    in_put = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
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
    dp1 = dropout(fc1, 0.5)
    fc2 = fully_connected(dp1, 4096, activation='relu')
    dp2 = dropout(fc2, 0.5)
    fc3 = fully_connected(dp2, num_classes, activation='softmax', restore=False)
#     network = fc3
    network = regression(fc3, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)
    return network


def fine_tune_Alexnet(network, X, Y, save_model_path, fine_tune_model_path):
    # Training
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output_RCNN')
    if os.path.isfile(fine_tune_model_path + '.index'):
        print("Loading the fine tuned model")
        model.load(fine_tune_model_path)
    elif os.path.isfile(save_model_path + '.index'):
        print("Loading the alexnet")
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False

    model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_rcnnflowers2')
    # Save the model
    model.save(fine_tune_model_path)


if __name__ == '__main__':
    data_set = config.FINE_TUNE_DATA
    if len(os.listdir(config.FINE_TUNE_DATA)) == 0:
        print("Reading Data")
        prep.load_train_proposals(config.FINE_TUNE_LIST, config.FINE_TUNE_CLASS - 1, save=True, save_path=data_set)
    print("Loading Data")
    X, Y = prep.load_from_npy(data_set)
    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print("Continue fine-tune")
    # three classes include background
    net = create_alexnet(config.FINE_TUNE_CLASS, restore=restore)
    fine_tune_Alexnet(net, X, Y, config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)
