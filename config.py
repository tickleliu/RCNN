# config path and files
import os
IMAGE_SIZE = 227
SAVE_MODEL = './pre_train_model'
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL, 'alexnet_imagenet.model')
FINE_TUNE_MODEL = './fine_tune_model'
FINE_TUNE_MODEL_PATH = os.path.join(FINE_TUNE_MODEL, 'fine_tune_model_save.model')
TRAIN_LIST = './train_list.txt'
FINE_TUNE_LIST = './fine_tune_list.txt'
FINE_TUNE_DATA = './data_set'
TRAIN_SVM = './svm_train'
TRAIN_SVR = './svr_train'
RESULT= './result'
TRAIN_CLASS = 2
FINE_TUNE_CLASS = 2 