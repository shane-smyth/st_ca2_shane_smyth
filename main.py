import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from image_processing import display_augmented_and_original, batch_generator

num_bins = 25
samples_per_bin = 500
data_dirs = [
    "./track1",
    "./track2"
]

def main():
  data = load_data()
  bins, centre = bin_and_plot_data(data)
  balanced_data = balance_data(data, bins)
  plot_balanced_data(balanced_data, centre)
  X_train, X_valid, y_train, y_valid, image_paths = split_data(balanced_data)
  plot_validation_training_distribution(y_train, y_valid)
  display_augmented_and_original(image_paths, y_train)
  model = nvidia_model()
  train_and_test_model(model, X_train, y_train, X_valid, y_valid)


def  plot_validation_training_distribution(y_train, y_valid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].set_title('Training data')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].set_title('Validation data')
    plt.show()  
    
    
def train_and_test_model(model, X_train, y_train, X_valid, y_valid):
    print(model.summary())
    history = model.fit(
      batch_generator(X_train, y_train, 100, 1), 
      steps_per_epoch=300, 
      epochs=20, 
      validation_data=batch_generator(X_valid, y_valid, 300, 0), 
      validation_steps=200, 
      verbose=1,
      shuffle=1,
      )
    model.save('basic_augmented_nvidia_model.keras')
    # this is a regression problem so accuracy is not an appropriate metric
    # you would have to get the steering angle exactly right for it to be
    # conunted as accurate. As there is an infinite number of possible steering angles,
    # this is almost imposible. Plot the loss instead
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    optimizer = Adam(learning_rate = 0.001)
    model.compile(loss='mse', optimizer = optimizer)
    return model

  
def split_data(data):
    image_paths, steerings = load_steering_img(data)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=77)
    print(f"Training samples {len(X_train)}, Validation samples {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid, image_paths
  

def load_steering_img(data):
    image_path, steerings = [], []
    for i in range(len(data)):
        row = data.iloc[i]
        for cam in ['center', 'left', 'right']:
            image_path.append(row[cam])
            steerings.append(float(row['steering']))
    return np.asarray(image_path), np.asarray(steerings)  
  
  
def plot_balanced_data(balanced_data, centre):
    hist, _ = np.histogram(balanced_data['steering'], num_bins)
    plt.bar(centre, hist, width=0.05)
    plt.plot((np.min(balanced_data['steering']), np.max(balanced_data['steering'])), (samples_per_bin, samples_per_bin))
    plt.show()


def balance_data(data, bins):
    # Too many zeros, this would bias the model to pretty much always drive straight. This would be dangerous
    remove_list = []
    for i in range(num_bins):
        list_ = []
        for j in range(len(data['steering'])):
            if bins[i] <= data['steering'][j] <= bins[i+1]:
                list_.append(j)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    print("Remove: ", len(remove_list))
    data = data.drop(data.index[remove_list])
    print("Remaining: ", len(data))
    return data


def bin_and_plot_data(data):
    hist, bins = np.histogram(data['steering'], num_bins)
    print(bins)
    centre = (bins[:-1] + bins[1:])*0.5
    plt.bar(centre, hist, width=0.05)
    plt.show()
    return bins, centre


def load_data():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    frames = []
    for data_dir in data_dirs:
        df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)
        img_dir = os.path.join(data_dir, 'IMG')
        for cam in ['center', 'left', 'right']:
            df[cam] = df[cam].apply(lambda p: os.path.join(img_dir, path_leaf(p).strip()))
        df["data_dir"] = os.path.basename(data_dir)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(data['data_dir'].value_counts())
    return data


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


if __name__ == "__main__":
  main()