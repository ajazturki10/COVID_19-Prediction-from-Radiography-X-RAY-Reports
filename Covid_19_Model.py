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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')


#Image Folder
IMAGES_PATH = 'D:/COVID_19_Radiography/'

covid_dir = os.path.join(IMAGES_PATH, 'COVID')
normal_dir = os.path.join(IMAGES_PATH, 'Normal')


#Plotting Digit
def plot_digit_with_label(img, label):
    plt.figure(figsize = (5, 5))
    if label == 0:
        print('Label : NORMAL')
    else:
        print('Label : COVID')
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()
    
#Get the images and labels   
def get_data(image_dir, label = None):
    images_labels = []

    for img in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        
        if label == 'COVID':
            images_labels.append((image, 1))
        else:
            images_labels.append((image, 0))
    return images_labels   

#Split the Images, labels
def split_images_labels(images_labels):
    images, labels = [], []
    
    for img, label in images_labels:
        images.append(img)
        labels.append(label)
    return images, labels


covid_images_labels = get_data(covid_dir, 'COVID')  #Covid Images
normal_images_labels = get_data(normal_dir, 'NORMAL') #Normal(Non-Covid) Images

images_labels = np.vstack((covid_images_labels, normal_images_labels))  #Combining both the types of images
images_labels = shuffle(shuffle(shuffle(images_labels)))  #Random Shuffling

images, labels = split_images_labels(images_labels) 

#plot_digit_with_label(images[11], labels[11])  uncomment and see the plots
#plot_digit_with_label(images[0], labels[0])

def scale_images(images):
    return images = np.array(images).reshape(-1, 64, 64, 1) / 255.0

images = scale_images(images)
labels = np_utils.to_categorical(labels)


def train_test_split(X, y, train_size = 0.7, test_size = 0.15):
    size = len(X)
    
    X_train = X[ : int((train_size * size))]
    X_val = X[int((train_size * size)) : int(((train_size + test_size) * size))]
    X_test = X[int(((train_size + test_size) * size)) : size]
    

    y_train = y[ : int((train_size * size))]
    y_val = y[int((train_size * size)) : int(((train_size + test_size) * size))]
    y_test = y[int(((train_size + test_size) * size)) : size]
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

(X_train, X_val, X_test), (y_train, y_val, y_test) = train_test_split(images, labels)


#Define the Model
def covid_model():
    # define model
    model = keras.models.Sequential()
    #Convolve block
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'tanh',
                                  kernel_initializer = 'glorot_uniform', input_shape=(64, 64, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'tanh', 
                                  kernel_initializer = 'glorot_uniform'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(keras.layers.Conv2D(128, kernel_size = (3, 3), activation = 'tanh', 
                                 kernel_initializer = 'glorot_uniform'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'tanh', 
                                kernel_initializer = 'glorot_uniform'))
    
    #Output layer
    model.add(keras.layers.Dense(2, activation = 'sigmoid'))
    
    optimizer = keras.optimizers.RMSprop()
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model

#call the model
model = covid_model()

#Training
model.fit(X_train, y_train, epochs = 25,
         validation_data = (X_val, y_val), batch_size = 256)


model.evaluate(X_test, y_test)
#[0.2129719853401184, 0.9469112157821655


#Save the Model
model.save(os.path.join(IMAGES_PATH, 'COVID_19.h5'))


#Evaluation
print("PRECISION : ", round(precision_score(preds.argmax(axis = 1), y_test.argmax(axis = 1)), 3))
# -->PRECISION :  0.899
print('RECALL : ', round(recall_score(preds.argmax(axis = 1), y_test.argmax(axis = 1)), 3))
# -->RECALL :  0.905
print('F1-SCORE : ',round(f1_score(preds.argmax(axis = 1), y_test.argmax(axis = 1)), 3))
# -->F1-SCORE :  0.902

#AUC Score
print(roc_auc_score(preds.argmax(axis = 1), y_test.argmax(axis = 1)))
# --> 0.933

#Plotting ROC Curve
fpr, tpr, threshold = roc_curve(y_test.argmax(axis = 1), preds.argmax(axis = 1))

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])  
    plt.annotate('AUC SCORE = {}%'.format(round(roc_auc_score(preds.argmax(axis = 1), y_test.argmax(axis = 1)), 3)),
            xy = (0.2, 0.82))# Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)  

#plot_roc_curve(fpr, tpr, )

#Thank You



    

