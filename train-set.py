import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

DATADIR = "./train_images/"
CATEGORIES = ["dog", "cat"]
IMG_SIZE = 100

'''for category in CATEGORIES:  # do dog and cat
    path = os.path.join(DATADIR,category)  # create path to dog and cat
    for img in os.listdir(path):  # iterate over each image per dog and cat
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        IMG_SIZE = 50
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        # plt.imshow(img_array, cmap='gray')  # graph it
        # plt.show()  # display!
        # break  # we just want one for now so break
    # break  #...and one more!'''

training_data = []
def create_training_data():
    for category in CATEGORIES:  # do dog and cat

        path = os.path.join(DATADIR,category)  # create path to dog and cat
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dog and cat
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
# We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

print("finish")
