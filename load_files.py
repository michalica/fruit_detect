import cv2
import os
import numpy as np
from random import sample


def load_random_pictures_from_dir(directory, number):
    prefix = directory

    for subbDir in os.listdir(prefix):
        print(subbDir)
        index = 0
        images = []
        number_of_fruits = len(os.listdir(prefix + subbDir))
        indexes = sample(range(0, number_of_fruits - 1), number)
        for file in os.listdir(prefix + subbDir):
            print(prefix + subbDir + '/' + file)
            img = cv2.imread(prefix + subbDir + '/' + file)

            if index in indexes:
                images.append(img)
            index += 1

            if len(images) == number:
                break

        both = np.hstack(images)
        cv2.imshow(subbDir, both)
        cv2.waitKey()
