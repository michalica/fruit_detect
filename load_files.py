import cv2
import os
import numpy as np

def load_random_pictures_from_dir(directory, number):
    prefix = directory

    for subbDir in os.listdir(prefix):
        print(subbDir)
        index = 0
        images = []
        for file in os.listdir(prefix + subbDir):
            print(prefix + subbDir + '/' + file)
            img = cv2.imread(prefix + subbDir + '/' + file)
            images.append(img)
            index += 1
            if index > number:
                break

        both = np.hstack((images[0], images[1], images[2], images[3]))
        cv2.imshow(subbDir, both)
        cv2.waitKey()