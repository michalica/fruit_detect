import os
import cv2
from random import sample
from load_files import load_random_pictures_from_dir
from create_pickle import merge_datasets
#from create_pickle import merge_datasets_tests
from create_pickle import create_pickle
from create_pickle import verify
import numpy as np
from PIL import Image
from six.moves import cPickle as pickle
from create_pickle import order_of_fruit

PIXEL_DEPTH = 255
IMAGE_SIZE = 100

NUMBER_OF_SAMPLES = 5
PATH_TO_TRAIN_DATASET = "dataset/Training/"
PATH_TO_TEST_DATASET = "dataset/Test/"
NUMBER_OF_TRAIN_DATA = 15000
NUMBER_OF_VALID_DATA = 6000
NUMBER_OF_TEST_DATA = 6000
NUMBER_OF_DATA = 21456


def main():
    # Uloha 1
    # load_random_pictures_from_dir(PATH_TO_TRAIN_DATASET, NUMBER_OF_SAMPLES)
    # Úloha 2
    train_data = create_pickle(PATH_TO_TRAIN_DATASET)
    #test_data = create_pickle(PATH_TO_TEST_DATASET)
    #
    # verify(test_data)
    # verify(train_data)
    #
    train_dataset, train_labels, valid_dataset, valid_labels = merge_datasets(train_data, NUMBER_OF_TRAIN_DATA, NUMBER_OF_VALID_DATA)
    #test_dataset, test_labels, = merge_datasets_tests(test_data, NUMBER_OF_TEST_DATA)

    print('dsads')

    # lowest = 500
    # for folder in os.listdir(PATH_TO_TRAIN_DATASET):
    #     if len(folder.split(".")) > 1:
    #         continue
    #     if len(os.listdir(PATH_TO_TRAIN_DATASET + '/' + folder)) < lowest:
    #         lowest = len(os.listdir(PATH_TO_TRAIN_DATASET + '/' + folder))
    #     print(len(os.listdir(PATH_TO_TRAIN_DATASET + '/' + folder)))
    #
    # print("Lowest number")
    # print(lowest)

if __name__ == '__main__':
    main()
