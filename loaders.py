import numpy as np
from musket_core.datasets import PredictionItem
from PIL import Image
import pandas as pd
import os
DIR="D:/cells"


class ProteinDataGenerator:

    def __init__(self, paths, labels, glued_images=True):
        self.paths = paths
        self.labels = labels
        self.glued_images = glued_images

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X,y = self.__load_image(self.paths[idx]),self.labels[idx]
        return PredictionItem(self.paths[idx],X, y)

    def __load_image(self, path):
        if self.glued_images:
            im = np.array(
                Image.open(path + '.png')
            )
        else:
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

class ProteinDataGeneratorClazz:

    def __init__(self, paths, labels, clazz, glued_images):
        self.paths, self.labels = paths, labels
        self.clazz=clazz
        self.glued_images = glued_images

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X,y = self.__load_image(self.paths[idx]),self.labels[idx]
        y1=np.array([y[self.clazz]])
        return PredictionItem(self.paths[idx],X, y1)

    def isPositive(self, item):
        v=self.labels[item]

        return v[self.clazz]==1

    def __load_image(self, path):
        if self.glued_images:
            im = np.array(
                Image.open(path + '.png')
            )
        else:
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
    
def getTrainDataset(data_path=DIR, images_dir='train', pd_file='train.csv'):
    path_to_train = f"{DIR}/{images_dir}/"
    data = pd.read_csv(f"{DIR}/{pd_file}")
    paths = []
    labels = []
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(paths), np.array(labels)


def getTrainDatasetForClass(clazz):
    path_to_train = DIR + '/train/'
    data = pd.read_csv(DIR + '/train.csv')
    paths = []
    labels = []
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(1)
        for key in lbl:
            if key==clazz:
                y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(paths), np.array(labels)

def getTrainDataset2(data_path=DIR, images_dir='train2', pd_file='train2.csv'):
    path_to_train = f"{DIR}/{images_dir}/"
    data = pd.read_csv(f"{DIR}/{pd_file}")
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