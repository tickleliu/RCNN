import os
import shutil
import config
import argparse
import sys


def svr():
    print('delete bounding box model and npy data \n')
    shutil.rmtree(os.path.join(config.TRAIN_SVR, '1'))
    os.mkdir(os.path.join(config.TRAIN_SVR, '1'))
    for file in os.listdir(config.TRAIN_SVR):
        if file.split('.')[-1] == 'pkl':
            os.remove(os.path.join(config.TRAIN_SVR, file))


def svm():
    print('delete svm model and npy data \n')
    shutil.rmtree(os.path.join(config.TRAIN_SVM, '1'))
    os.mkdir(os.path.join(config.TRAIN_SVM, '1'))
    for file in os.listdir(config.TRAIN_SVM):
        if file.split('.')[-1] == 'pkl':
            os.remove(os.path.join(config.TRAIN_SVM, file))


def finetune():
    print('delete finetune model and npy data \n')
    shutil.rmtree(config.FINE_TUNE_DATA)
    os.mkdir(config.FINE_TUNE_DATA)
    shutil.rmtree(config.FINE_TUNE_MODEL)
    os.mkdir(config.FINE_TUNE_MODEL)


def pretrain():
    print('delete pretrain model \n')
    shutil.rmtree(config.SAVE_MODEL)
    os.mkdir(config.SAVE_MODEL)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--pretrain", type=int, default=0, help='delete pretrain model')
    parse.add_argument("--finetune", type=int, default=0, help='delete finetune model')
    parse.add_argument("--svm", type=int, default=0, help='delete svm classify model')
    parse.add_argument("--bbr", type=int, default=0, help='delete bounding box regression model')
    arg = parse.parse_args(sys.argv[1:])
    if arg.pretrain == 1:
        pretrain()
    if arg.finetune == 1:
        finetune()
    if arg.svm == 1:
        svm()
    if arg.bbr == 1:
        svr()
