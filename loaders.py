import numpy as np
from musket_core.datasets import PredictionItem
from PIL import Image
import pandas as pd
import os
DIR="D:/cells"
class ProteinDataGenerator:

    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X,y = self.__load_image(self.paths[idx]),self.labels[idx]
        return PredictionItem(self.paths[idx],X, y)

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)
        ), -1)
        return im

def getTrainDataset():
    path_to_train = DIR + '/train/'
    data = pd.read_csv(DIR + '/train.csv')
    paths = []
    labels = []
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(paths), np.array(labels)

def getTrainDataset2():
    path_to_train = DIR + '/train2/'
    data = pd.read_csv(DIR + '/train2.csv')
    paths = []
    labels = []
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            i=int(key)
            if i<28:
               y[i] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(paths), np.array(labels)

def getTestDataset():
    path_to_test = DIR + '/test/'
    data = pd.read_csv(DIR + '/sample_submission.csv')
    paths = []
    labels = []
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)