import numpy as np
from musket_core.datasets import PredictionItem
from PIL import Image
import pandas as pd
import os

DIR =                os.getenv('MAIN_DIR', 'D:/cells')
GLUED_IMAGES =       os.getenv('GLUED_IMAGES', False)
TRAIN_SUBDIR =       os.getenv('TRAIN_SUBDIR', 'train')
EXTRA_TRAIN_SUBDIR = os.getenv('EXTRA_TRAIN_SUBDIR', 'train2')
TRAIN_CSV =          os.getenv('TRAIN_CSV', 'train.csv')
EXTRA_TRAIN_CSV =    os.getenv('EXTRA_TRAIN_CSV', 'train2.csv')
TEST_SUBDIR =        os.getenv('TEST_SUBDIR', 'test')

import musket_core.datasets as ds
class ProteinDataGenerator:

    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X,y = self.__load_image(self.paths[idx]),self.labels[idx]
        return PredictionItem(self.paths[idx],X, y)

    def __load_image(self, path):
        if GLUED_IMAGES:
            im = np.array(Image.open(path + '.png'))
        else:
            R = Image.open(path + '_red.png')
            G = Image.open(path + '_green.png')
            B = Image.open(path + '_blue.png')
            Y = Image.open(path + '_yellow.png')
            try:
                im = np.stack((
                    np.array(R),
                    np.array(G),
                    np.array(B),
                    np.array(Y)
                ), -1)
            except:
                return np.zeros((512,512,4))
        return im

class ProteinDataGeneratorClazz:

    def __init__(self, paths, labels,clazz):
        self.paths, self.labels = paths, labels
        self.clazz=clazz

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
    path_to_train = DIR + f'/{TRAIN_SUBDIR}/'
    data = pd.read_csv(DIR + f'/{TRAIN_CSV}')
    paths = []
    labels = []
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(paths), np.array(labels)

def calculate_fold_indexes(paths):
    foldSets = []
    foldIndexes = []
    for i in range(1, 6):
        train = pd.read_csv("./folds/fold" + str(i) + "_train.csv")
        test = pd.read_csv("./folds/fold" + str(i) + "_val.csv")
        foldSets.append((set(train["Id"].values), set(test["Id"].values)))
        foldIndexes.append([[], []])
    for i in range(len(paths)):
        bn = os.path.basename(paths[i])
        for j in range(len(foldSets)):
            if bn in foldSets[j][0]:
                foldIndexes[j][0].append(i)
            if bn in foldSets[j][1]:
                foldIndexes[j][1].append(i)
    return foldIndexes


def calculate_holdout_indexes(paths):
    train = pd.read_csv("./folds/holdout.csv" )
    fold=set(train["Id"].values)
    foldIndexes=[]
    for i in range(len(paths)):
        bn = os.path.basename(paths[i])
        if bn in fold:
            foldIndexes.append(i);
    return foldIndexes

def getTrainDatasetForClass(clazz):
    path_to_train = DIR + f'/{TRAIN_SUBDIR}/'
    data = pd.read_csv(DIR + f'/{TRAIN_CSV}')
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

def getTrainDataset2():
    path_to_train = DIR + f'/{EXTRA_TRAIN_SUBDIR}/'
    data = pd.read_csv(DIR + f'/{EXTRA_TRAIN_CSV}')
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

def get_test_paths_and_ids():
    path_to_test = DIR + f'/{TEST_SUBDIR}/'
    data = pd.read_csv(DIR + '/sample_submission.csv')
    paths = []
    labels = []
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def createDataSet():
    paths, labels = getTrainDataset()
    paths2, labels2 = getTrainDataset2()
    paths = np.concatenate([paths2, paths])
    labels = np.concatenate([labels2, labels])
    foldIndexes = calculate_fold_indexes(paths)
    tg = ProteinDataGenerator(paths, labels)
    tg.folds = foldIndexes;
    return tg

def createHoldoutDataSet():
    paths, labels = getTrainDataset()
    paths2, labels2 = getTrainDataset2()
    paths = np.concatenate([paths2, paths])
    labels = np.concatenate([labels2, labels])
    foldIndexes = calculate_fold_indexes(paths)
    tg = ProteinDataGenerator(paths, labels)
    tg.folds = foldIndexes;
    hi = calculate_holdout_indexes(paths)
    test = ds.SubDataSet(tg, hi)
    return test,labels[hi]

def getSubmitSample():
    return pd.read_csv(DIR + '/sample_submission.csv')


def getTestDataSet():
    pathsTest, labelsTest = get_test_paths_and_ids()
    return ProteinDataGenerator(pathsTest, labelsTest)
