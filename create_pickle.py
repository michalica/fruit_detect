import os
import os.path
import numpy as np
from scipy import ndimage
import six
from six.moves import cPickle as pickle
import math
import cv2
import random
from PIL import Image


PIXEL_DEPTH = 255
IMAGE_SIZE = 100
NUMBER_OF_FOLDERS = 48
order_of_fruit = []
VALID_TRAIN_SIZE = 6000
NUMBER_OF_TRAIN_DATA = 15000


def load_images(folders, test=False):
    if test:
        size = int(VALID_TRAIN_SIZE / NUMBER_OF_FOLDERS)
    else:
        size = int(NUMBER_OF_TRAIN_DATA / NUMBER_OF_FOLDERS)
    counter = 0

    dataset = np.ndarray(shape=(size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    for images in folders:
        for image in os.listdir(images):
            counter += 1
            image_file = os.path.join(images, image)
            print(image_file)

            if counter == size:
                break
            try:
                data = (ndimage.imread(image_file).astype(float) / PIXEL_DEPTH) - 0.5
                if data.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    raise Exception('Bad shape of image')
                dataset[:, :, :, :] = data
            except IOError:
                print('Failed load image ' + image_file)

    return dataset


def create_pickle(directory, test=False):
    prefix = directory
    print('========================================')
    print('Start creating pickle files')
    already_processed = []
    datasets = []
    for subbDir in os.listdir(prefix):
        if os.path.isfile(prefix + subbDir):
            continue

        if len(already_processed) > 0:
            if subbDir in already_processed:
                continue

        already_processed.append(subbDir)
        print(subbDir)
        first_word = subbDir.split(" ")[0]
        pickle_file = directory + first_word + '.pickle'

        if pickle_file not in datasets:
            datasets.append(pickle_file)

        if os.path.exists(pickle_file):
            order_of_fruit.append(first_word)
            print('****************************************')
            print('Pickle file %s already exist' % pickle_file)
            print('****************************************')
        else:
            print('****************************************')
            print('Pickling ' + first_word)
            allFiles = []
            allFiles.append(directory + '/' + subbDir)
            order_of_fruit.append(first_word)
            for tmpSubDir in os.listdir(prefix):
                tmpSubDir.split(" ")
                if subbDir == tmpSubDir:
                    continue
                if first_word == tmpSubDir.split(" ")[0]:
                    allFiles.append(directory + '/' + tmpSubDir)
                    already_processed.append(tmpSubDir)

            if test:
                group_of_images = load_images(allFiles, True)
            else:
                group_of_images = load_images(allFiles)

            try:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(group_of_images, f, pickle.HIGHEST_PROTOCOL)
                    print('Number of images - image_size - image_size:', group_of_images.shape)
                    print('****************************************')
                    print(group_of_images)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
        print('****************************************')
    print('Finish creating pickle files')
    print('========================================')
    print(len(already_processed))
    print(datasets[1])
    return datasets


def verify(pic_file):
    collection = []
    for x in range(10):
        with open(pic_file[x], 'rb') as f:
            letter_set = pickle.load(f)
            rand = random.randint(0, len(letter_set))
            image = letter_set[rand, :, :]
            if len(collection) == 0:
                    collection = image
            else:
                collection = np.concatenate((collection, image), axis=1)
    cv2.imshow('Verification images', collection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def create_dataset_and_labels(size):
    if size > 0:
        print('========================================')
        print('Creating dataset and labels')
        print('========================================')
        dataset = np.ndarray(shape=(size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        labels = np.ndarray(size, dtype=np.int32)
    else:
        print('Merging test dataset')
        dataset, labels = None, None

    return dataset, labels



# def merge_datasets(pickle_files, train_size, valid_size):
#     print('========================================')
#     print('Merging datasets start.')
#     valid_dataset, valid_labels = create_dataset_and_labels(valid_size)
#     train_dataset, train_labels = create_dataset_and_labels(train_size)
#     valid_size_letter = int(valid_size / NUMBER_OF_FOLDERS)
#     train_size_letter = int(train_size / NUMBER_OF_FOLDERS)
#
#     # valid_start, valid_end = 0, valid_size_letter
#     # train_start, train_end = 0, train_size_letter
#     # end_letter = valid_size_letter + train_size_letter
#     lowest = 500
#     print(pickle_files)
#     for label, file in enumerate(pickle_files):
#         try:
#             with open(file, 'rb') as load_file:
#                 char = pickle.load(load_file)
#                 np.random.shuffle(char)
#                 print(file)
#                 valid_dataset[0:valid_size_letter, :, :, :] = char[:valid_size_letter, :, :, :]
#                 valid_labels[0:valid_size_letter] = label
#                 train_end = valid_size_letter + train_size_letter
#
#                 train_dataset[valid_size_letter:train_end, :, :, :] = char[valid_size_letter:train_end, :, :, :]
#                 train_labels[valid_size_letter:train_end] = label
#
#         except Exception as e:
#             print('Failed load pickle: ' + file)
#             print(e)
#
#     print("lowest")
#     print(lowest)
#     print('Merging datasets finished.')
#     print('========================================')
#     return train_dataset, train_labels, valid_dataset, valid_labels

def merge_datasets(pickle_files, train_size, valid_size):
    print('========================================')
    print('Merging datasets start.')
    valid_dataset, valid_labels = create_dataset_and_labels(valid_size)
    train_dataset, train_labels = create_dataset_and_labels(train_size)
    valid_size_letter = int(valid_size / NUMBER_OF_FOLDERS)
    train_size_letter = int(train_size / NUMBER_OF_FOLDERS)

    valid_start, valid_end = 0, valid_size_letter
    train_start, train_end = 0, train_size_letter
    end_letter = valid_size_letter + train_size_letter
    for label, file in enumerate(pickle_files):
        try:
            with open(file, 'rb') as load_file:
                char = pickle.load(load_file)
                np.random.shuffle(char)
                if valid_dataset is not None:
                    valid_dataset[valid_start:valid_end, :, :, :] = char[:valid_size_letter, :, :, :]
                    valid_labels[valid_start:valid_end] = label
                    valid_start, valid_end = valid_start + valid_size_letter, valid_end + valid_size_letter

                train_dataset[train_start:train_end, :, :, :] = char[valid_size_letter:end_letter, :, :, :]
                train_labels[train_start:train_end] = label
                train_start, train_end = train_start + train_size_letter, train_end + train_size_letter
        except Exception:
            print('Failed load pickle: ' + file)

    print('Merging datasets finished.')
    print('========================================')
    return train_dataset, train_labels, valid_dataset, valid_labels

# def merge_datasets_tests(pickle_files, test_size):
#     print('========================================')
#     print('Merging datasets start.')
#     test_dataset, test_labels = create_dataset_and_labels(test_size)
#     test_size_letter = int(test_size / NUMBER_OF_FOLDERS)
#     testing_labels = []
#     testing_labels_2 = []
#     test_end = 0
#     # valid_start, valid_end = 0, valid_size_letter
#     # train_start, train_end = 0, train_size_letter
#     # end_letter = valid_size_letter + train_size_letter
#     lowest = 500
#     print(pickle_files)
#     for label, file in enumerate(pickle_files):
#         try:
#             with open(file, 'rb') as load_file:
#                 char = pickle.load(load_file)
#                 np.random.shuffle(char)
#                 print(file)
#                 img = Image.fromarray(char[0, :, :, :])
#                 img.save('my.png')
#                 img.show()
#
#
#
#         except Exception as e:
#             print('Failed load pickle: ' + file)
#             print(e)
#
#     print("lowest")
#     print(lowest)
#     print('Merging datasets finished.')
#     print('========================================')
#     return test_dataset, testing_labels


