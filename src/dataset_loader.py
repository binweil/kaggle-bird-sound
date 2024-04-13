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

class DataSetsLoader:
    def __init__(self, num_class=10, use_internet=True):
        self.num_class = num_class
        self.data_dir = ""
        self.use_internet = use_internet
        self.load_datasets()

    def load_datasets(self):
        folder = os.getcwd()
        self.data_dir = os.path.join(folder, "data/train_audio")

    def scale_minmax(self, X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled
        
    def get_log_mel_spectrogram(self, data_path):
        # load audio files with librosa
        scale, sample_rate = librosa.load(data_path, duration=10)
        # n_fft: frame rate, sr: sampleing rate, n_mels: number of mel bands
        mel_spectrogram = librosa.feature.melspectrogram(y=scale, n_fft=2048, sr=sample_rate, n_mels=10, hop_length=1024, fmin=1000, fmax=sample_rate/2, htk=True)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram**3, ref=np.max)
        return log_mel_spectrogram

    def process_files(self, data_inputs):
        bird_class = os.path.basename(data_inputs[0])
        # print("processing bird class: {}".format(data_inputs[0]))
        index = data_inputs[1]
        class_map = data_inputs[6]
        class_map[index] = bird_class;
        
        total_files = glob.glob(os.path.join(self.data_dir, os.path.basename(bird_class) + "/*"))
        total_file_number = len(total_files)
        
        saved_image_path = ""
        log_mel_spectrogram = []
        for file in total_files:
            train_data_path = os.path.join(os.getcwd(), "data/" + "train_data/" + bird_class)
            if not os.path.exists(train_data_path):
                os.makedirs(train_data_path)
            saved_image_path = os.path.join(os.getcwd(), "data/" + "train_data/" + bird_class + "/" + os.path.basename(file))
            saved_image_path = saved_image_path.replace(".ogg", ".png")
            
            log_mel_spectrogram = self.get_log_mel_spectrogram(file)
            log_mel_spectrogram_img = cv2.resize(log_mel_spectrogram, dsize=(255, 255), interpolation=cv2.INTER_AREA)
            cv2.imwrite(saved_image_path, log_mel_spectrogram_img.astype(np.uint8))
        
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
        pool = Pool(1)
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