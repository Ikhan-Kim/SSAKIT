from os import listdir
# from os.path import isfile, join
import os
import sys
import shutil
from distutils.dir_util import copy_tree
import random


def set_directory(dataset_name, class_name, copy_path):
    train_path = "./learnData/" + dataset_name + "/train/" + class_name + '/'
    validation_path = "./learnData/" + dataset_name + "/validation/" + class_name + '/'
    test_path = "./learnData/" + dataset_name + "/test/" + class_name + '/'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    copy_tree(copy_path, train_path)
    file_list = os.listdir(train_path)
    file_len = int(len(file_list) * 0.2)
    for file_name in random.sample(file_list, file_len):
        shutil.move(train_path + file_name, validation_path)
    file_list = os.listdir(train_path)
    for file_name in random.sample(file_list, file_len):
        shutil.move(train_path + file_name, test_path)


def name():
    print('set_directory')