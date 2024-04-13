import os
import sys
import glob

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import librosa
import librosa.display
import IPython.display as ipd

from multiprocessing import Process
from multiprocessing import Pool, Manager

import skimage.io
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageDatasetsLoader:
    def __init__(self, num_class=10):
        self.num_class = num_class
        self.data_dir = ""
        self.load_datasets()

    def load_datasets(self):
        folder = os.getcwd()
        self.data_dir = os.path.join(folder, "data/train_data")

    def load_images(self, image_path):
        img = cv2.imread(image_path)
        return img
        
    
    def process_files(self, data_inputs):
        folder = data_inputs[0]
        index = data_inputs[1]
        train_sounds = data_inputs[2]
        train_labels = data_inputs[3]
        test_sounds = data_inputs[4]
        test_labels = data_inputs[5] 
        class_map = data_inputs[6]

        print("processing class: {}".format(os.path.basename(folder)))
        
        class_map[index] = os.path.basename(folder);
        # gather all files
        total_files = glob.glob(os.path.join(self.data_dir, os.path.basename(folder) + "/*"))
        total_file_number = len(total_files)

        class_train_files = []
        class_train_labels = []
        class_test_files = []
        class_test_labels = []
        for f in total_files:
            if len(class_train_files) < total_file_number*0.7:
                class_train_files.append(self.load_images(f))
                class_train_labels.append(index)
            else:
                class_test_files.append(self.load_images(f))
                class_test_labels.append(index)
        train_sounds.extend(class_train_files)
        train_labels.extend(class_train_labels)
        test_sounds.extend(class_test_files)
        test_labels.extend(class_test_labels)
        
        print("processed class: {}".format(os.path.basename(folder)))
        
    def transform_to_tensorflow_dataset(self):
        train_sounds = Manager().list()
        train_labels = Manager().list()
        test_sounds = Manager().list()
        test_labels = Manager().list()
        class_map = Manager().dict()
        folders = glob.glob(os.path.join(self.data_dir, "[!README]*"))

        data_inputs = []
        for index, folder in enumerate(folders):
            data_inputs.append((
                folder, 
                index,
                train_sounds, 
                train_labels, 
                test_sounds, 
                test_labels, 
                class_map))
        
        print("Using {} number of CPU to process training data".format(os.cpu_count()))
        pool = Pool(10)
        pool.map(self.process_files, data_inputs)

        return (
            np.array(list(train_sounds)),
            np.array(list(test_sounds)),
            np.array(list(train_labels)),
            np.array(list(test_labels)),
            class_map,
        )


if __name__ == "__main__":
    datasets_loader = DataSetsLoader()
    # Show a example dataset
    # datasets_loader.show_sample_data()
    # Get the tensorflow compatiable dataset
    datasets_loader.transform_to_tensorflow_dataset()