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

num_bins = 25
samples_per_bin = 500
data_dir = "./training_data"

def main():
  data = load_data()
  bins, centre = bin_and_plot_data(data)
  balanced_data = balance_data(data, bins)
  plot_balanced_data(balanced_data, centre)
  
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
    data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names = columns)
    pd.set_option('display.width', None)
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    return data


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


if __name__ == "__main__":
  main()