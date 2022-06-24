
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import numpy as np
import audtorch
sample_rate = 16000

class AudioSet(Dataset):
    def __init__(self, file_path, label_path):
        # self.file_names = os.listdir(file_path)
        with open(label_path, 'r') as df:
            self.labels = df.readlines()
        self.label_folder = label_path

        self.file_folder = file_path
        self.transform = audtorch.transforms.RandomCrop(250, axis=-2)



    def __getitem__(self, index):
        labels = self.labels[index].split(',')
        file_name = labels[0][1:-1] + '.npy'

        signal = np.load(self.file_folder + file_name)
        if signal.shape[0] == 2:
            signal = signal.mean(0, keepdims=True)
        if self.transform is not None:
            signal = self.transform(signal)

        # data = np.array(data, dtype=np.float32)
        Subject_ID = labels[2]
        Country = float(labels[3])
        Country_string = labels[4]
        Age = int(float(labels[5]))

        Emotions = []

        for i in range(6, 15):
            Emotions.append(float(labels[i]))

        Emotions.append(float(labels[15].replace('\n', '')))
        Emotions = np.array(Emotions)

        return signal, Subject_ID, Country, Country_string, Age, Emotions

    def __len__(self):
        return len(self.labels)
