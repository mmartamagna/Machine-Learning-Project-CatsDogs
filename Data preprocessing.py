###Import packages
import pandas as pd #reading/writing data
import numpy as np #working with arrays, linear algebra
import matplotlib.pyplot as plt #display images
import matplotlib.image as mpimg
import os
from os import listdir
import seaborn as sns
import random
import pickle

#Image preprocessing
import pathlib
import PIL
import PIL.Image
import cv2

#Neural Networks
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical, load_img, img_to_array
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, callbacks, regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Activation
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import binary_crossentropy
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

#Cross validation
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, zero_one_loss

tfk = tf.keras
#tf.keras.backend.set_floatx("float64")
%load_ext tensorboard

*******************************************************************
##DATA IMPORT
#Open the data from Google Drive
from google.colab import drive
drive.mount('/content/drive')
dir = '/content/drive/MyDrive/Project CatsDogs/CatsDogs'

#create two variables containing the data of cats and dogs
cats_folder = os.path.join(dir, 'Cats')
dogs_folder = os.path.join(dir, 'Dogs')

***********************************************************************
##DATA PREPROCESSING
##1. WRONG IMAGES REMOVAL
##Define vectors containing misleading images
wrong_cats = [666, 835, 1450, 2939, 3216, 3330, 3672, 3822, 4085, 4104, 4338, 4688, 4833, 5351,
              5355, 5418, 5583, 5673, 7377, 7564, 7920, 7968, 8456, 8470, 9171, 9770, 10404,
              10712, 10863, 11184, 11565, 12272]

wrong_dogs = [7, 1043, 1194, 1259, 1308, 1773, 1895, 2614, 2877, 4367, 5490, 5604, 5694, 6272,
              6475, 7798, 8507, 8736, 8898, 9188, 9517, 10161, 10190, 10237, 10401, 11702, 10747,
              10797, 10801, 11186, 11299, 11731, 12005, 12027, 12376]

#REMOVING ALL THE MISLEADING IMAGES
import shutil

for cat_number in wrong_cats:
    filename = f"{cat_number}.jpg"
    file_path = os.path.join(cats_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

# Remove misleading images from the dogs folder
for dog_number in wrong_dogs:
    filename = f"{dog_number}.jpg"
    file_path = os.path.join(dogs_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

 # Count the remaining images in the cats folder
cats_images = [file for file in os.listdir(cats_folder) if file.endswith(".jpg")]
num_cats_images = len(cats_images)

# Count the remaining images in the dogs folder
dogs_images = [file for file in os.listdir(dogs_folder) if file.endswith(".jpg")]
num_dogs_images = len(dogs_images)

print(f"Number of remaining cat images: {num_cats_images}")
print(f"Number of remaining dog images: {num_dogs_images}")

##2. SET IMAGES TO GREYSCALE AND RESIZE (128X128)
#Recall directory
dir = '/content/drive/MyDrive/Project CatsDogs/CatsDogs'

WIDTH = 128
HEIGHT = 128
img_size = (WIDTH, HEIGHT)

#Set channel = 1 for grey images
channels = 1

# Define the list in which the images will be stored and categories of pets
pets_grey = []
categories = ['Cats','Dogs']

# Function to load the data, transform into array of grey images, assign data to a class and store them in the pets list
def create_data_grey():

# the function iterates throught the two sub-directories
    for category in categories:
        path = os.path.join(dir, category)
        pet_class = categories.index(category) # assign 0 to cat and 1 to dog, according to the index of the categories

# the function iterates through each image in both the folders. Opencv (cv2 when imported) package allows to read and load them.
        for img in os.listdir(path):   
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, img_size)  #resize the images 128x128
                pets_grey.append([new_array, pet_class]) #appends the array of the image, together with the class value (creating a sub-list of two values for each image), to the pets list.
            except:
                pass

#Call the function
create_data_grey()

# Shuffle the order of the images
random.seed(256)
random.shuffle(pets_grey)

# split the data into images and labels arrays
images_array_grey = []
labels_array_grey = []

for image, label in pets_grey:
    images_array_grey.append(image)
    labels_array_grey.append(label)

images_array_grey = np.array(images_array_grey).reshape(-1, WIDTH, HEIGHT, channels)

#Open a file called "X.pickle" write in binary mode ('wb'). It creates a file object pickle_out that will be used to write the data.
pickle_output_grey = open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/images_grey.pickle','wb')
#pickle.dump(X_images, pickle_out) writes the content of the X array to the pickle file using the pickle.dump() function.
pickle.dump(images_array_grey, pickle_output_grey)
pickle_output_grey.close()

#Open a file called "y.pickle" write in binary mode ('wb').
pickle_output_grey = open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/labels_grey.pickle','wb')
#pickle.dump(Y, pickle_out) writes the content of the Y array to the pickle file using the pickle.dump() function.
pickle.dump(labels_array_grey, pickle_output_grey)
pickle_output_grey.close()

#Open a file called "X.pickle" and read in binary mode ('wb').
pickle_input_grey = open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/images_grey.pickle','rb')
images_array_grey = pickle.load(pickle_input_grey)

#Open a file called "Y.pickle" and write in binary mode ('wb').
pickle_input_grey = open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/labels_grey.pickle','rb')
labels_array_grey = pickle.load(pickle_input_grey)

##Call the data in pickle format
images_array_grey = pickle.load(open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/images_grey.pickle','rb'))
labels_array_grey = pickle.load(open('/content/drive/MyDrive/Project CatsDogs/CatsDogs/Pickles/labels_grey.pickle','rb'))



