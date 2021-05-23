import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

#PATH
PATH = 'D:/COVID_19_Radiography/'

#Load the Model
model = keras.models.load_model(os.path.join(PATH, 'COVID_19.h5'))


#Preprocessing the new image
def predict(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(cv2.resize(img, (64, 64))).reshape(-1, 64, 64, 1) / 255.0
    
    #predictions
    prediction = model.predict(img)
    
    if prediction.argmax(axis = 1) == 0:
        print("RADIOGRAPHY REPORT : NORMAL")
    else:
        print("RADIOGRAPHY REPORT : COVID")
        
#Sample Prediction
img_dir = os.path.join(PATH, 'test_image.jpeg')

predict(img_dir)
#RADIOGRAPHY REPORT : COVID

