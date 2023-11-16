# Importing necessary libraries 
import os, warnings
from collections import Counter
import argparse

import cv2
from PIL import Image
import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics, model_selection

import tensorflow as tf
import keras
from keras import callbacks, layers, utils



def set_seed(seed=31415):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def set_plt_defaults():
    """Set Matplotlib defaults"""
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
        titleweight='bold', titlesize=18, titlepad=10)
    plt.rc('image', cmap='magma')
    warnings.filterwarnings("ignore") # to clean up output cells

def get_classes(PATH):
    names = [name.replace(' ', '_').split('_')[0] for name in os.listdir(PATH)]
    classes = Counter(names)  #returns dictionary
    print(f"Total number of images is {len(names)}")
    return classes

def show_classe_distribution(classes):
    plt.figure(figsize = (12,8))
    plt.title('Class Counts in Dataset')
    plt.bar(*zip(*classes.items()))
    plt.xticks(rotation='vertical')
    plt.show()


def get_path_class(PATH, classes):
    return {key: [os.path.join(PATH, name)
                  for name in os.listdir(PATH)
                  if name.replace(' ', '_').split('_')[0] == key]
                for key in classes.keys()}


def show_images(path_class):
    fig = plt.figure(figsize=(15, 15))
    for i, key in enumerate(path_class.keys()):
        img1 = Image.open(path_class[key][0]) 
        img2 = Image.open(path_class[key][1]) 
        img3 = Image.open(path_class[key][2]) 

        ax = fig.add_subplot(8, 9,  3*i + 1, xticks=[], yticks=[])
        ax.imshow(img1)
        ax.set_title(key)
        
        ax = fig.add_subplot(8, 9,  3*i + 2, xticks=[], yticks=[])
        ax.imshow(img2)
        ax.set_title(key)

        ax = fig.add_subplot(8, 9,  3*i + 3, xticks=[], yticks=[])
        ax.imshow(img3)
        ax.set_title(key)
    plt.show()


def show_scatter_plot(PATH):
    """Show scatter plot of image sizes."""
    size = [cv2.imread(os.path.join(PATH, name)).shape for name in os.listdir(PATH)]
    x, y, _ = zip(*size)
    fig = plt.figure(figsize=(12, 10))

    # scatter plot
    plt.scatter(x,y)
    plt.title("Image size scatterplot")

    # add diagonal red line 
    max_dim = max(max(x), max(y))
    plt.plot([0, max_dim],[0, max_dim], 'r')
    plt.show()


def process_img(img, size = (128,128)):
    """Resize and normalize image"""
    img = cv2.resize(img, size)     # resize image
    if img.max() > 0:               # avoid division by zero
        img = img / img.max()       # normalize to [0,1]
    return img


def X_Y_split(PATH):
    """Split images into X and labels into Y"""
    X, Y = [], []
    for name  in os.listdir(PATH):
        img = cv2.imread(os.path.join(PATH, name))
        if img is None:  # check if image is correctly loaded
            print(f"Image {name} could not be loaded.")
            continue
        X.append(process_img(img))
        Y.append(name.replace(' ', '_').split('_')[0])

    X = np.array(X)
    return X, Y


def split_train_valid(X, Y):
    le = sklearn.preprocessing.LabelEncoder()
    Y_le = le.fit_transform(Y)
    Y_cat = utils.to_categorical(Y_le, 23)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y_cat, test_size=0.285, stratify=Y_le)
    print(f"Images in each class in Test set: {np.sum(Y_test, axis =0)}")
    return X_train, X_test, Y_train, Y_test, le


def build_model(X_train):
    input_shape =  X_train[0].shape
    output_shape = 23

    model = keras.Sequential([
        layers.Conv2D(filters = 16, kernel_size = 3, input_shape = input_shape, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 32, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 64, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters = 128, kernel_size = 2, activation= 'relu', padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(500, activation = 'relu'),
        # layers.Dropout(0.2),
        layers.Dense(150, activation = 'relu'),
        # layers.Dropout(0.2),
        layers.Dense(output_shape, activation = 'softmax'),
    ])
    model.summary()
    return model


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
    print('Model is Compiled!')

def create_datagenerator(X_train):
    datagener = keras.preprocessing.image.ImageDataGenerator(
                    rotation_range= 20,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,)
    # fit data generator
    datagener.fit(X_train)
    return datagener


def fit_model(model, datagener, X_train, Y_train, X_test, Y_test):
    batch_size = 4
    epochs = 500

    model_path = 'cnn.hdf5'
    callbecks = [callbacks.EarlyStopping(monitor ='val_loss', patience = 20), 
                callbacks.ModelCheckpoint(filepath = model_path, save_best_only = True)]

    history = model.fit(
            datagener.flow(X_train, Y_train, batch_size=batch_size), 
            batch_size = batch_size, 
            steps_per_epoch = len(X_train) // batch_size,
            epochs = epochs,
            validation_data = (X_train, Y_train),
            callbacks = callbecks,
            verbose = 1)

    model.load_weights(model_path)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test set accuracy: {score[1]}")
    return history


def plot_confusion_matrix(model, x, y, le, plot_title = ''):
    y_pred = model.predict(x)                            # get predictions on x using model
    predicted_categories = tf.argmax(y_pred, axis=1)     # get index of predicted category
    true_categories = tf.argmax(y, axis=1)               # get index of true category
    # create confusion matrix using sklearn
    cm = metrics.confusion_matrix(true_categories, predicted_categories)
    # create DataFrame from the confusion matrix. We retrieve labels from LabelEncoder.
    df_cm = pd.DataFrame(cm, index = le.classes_ ,  columns = le.classes_)
    # divide each row to its sum in the DataFrame to get normalized output
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    
    plt.figure(figsize = (15,12))
    plt.title(plot_title)
    sns.heatmap(df_cm, annot=True)
    plt.show()

def plot_metrics(history):
    plt.figure(figsize = (15,12))     
    
    plt.subplot(211)  
    plt.title('Model Accuracy')  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')  
    plt.legend(['Generated Data', 'Original Train Data'], loc='best')  
        
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('Model Loss')  
    plt.ylabel('Categorical Crossentropy')  
    plt.xlabel('Epoch')  
    plt.legend(['Generated Data', 'Original Train Data'], loc='best')  
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Pollen Classification')
    parser.add_argument('--path', type=str, required=True, help='Path to the Kaggle dataset')
    args = parser.parse_args()
    PATH = args.path

    set_seed()
    set_plt_defaults()

    classes = get_classes(PATH)
    show_classe_distribution(classes)
    path_class = get_path_class(PATH, classes)
    show_images(path_class)
    show_scatter_plot(PATH)

    X, Y = X_Y_split(PATH)
    X_train, X_test, Y_train, Y_test, le = split_train_valid(X, Y)
    model = build_model(X_train)
    compile_model(model)
    datagener = create_datagenerator(X_train)
    history = fit_model(model, datagener, X_train, Y_train, X_test, Y_test)

    plot_confusion_matrix(model, X_train, Y_train, le, "Train set")
    plot_confusion_matrix(model, X_test,  Y_test, le,  "Test set")
    plot_metrics(history)


if __name__ == "__main__":
    main()
