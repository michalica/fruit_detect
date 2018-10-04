import os
import cv2
from load_files import load_random_pictures_from_dir

NUMBER_OF_SAMPLES = 5
PATH_TO_TRAIN_DATASET = "dataset/Training/"

def main():
    # Uloha 1
    load_random_pictures_from_dir(PATH_TO_TRAIN_DATASET, NUMBER_OF_SAMPLES)

if __name__ == '__main__':
    main()
