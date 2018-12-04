import numpy as np
from musket_core.datasets import PredictionItem
from classification_pipeline import classification
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score as off1

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

DIR = 'F:/cells'

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


def getOptimalT(lastFullValPred,lastFullValLabels):
    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1

    print(np.max(f1s, axis=0))
    print(np.mean(np.max(f1s, axis=0)))

    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
    print('Choosing threshold: ', T, ', validation F1-score: ', np.mean(np.max(f1s,axis=0)))
    print(T)
    return T, np.mean(np.max(f1s, axis=0))


def main():
   paths, labels = getTrainDataset()
   pathsTest, labelsTest = getTestDataset()
   tg = ProteinDataGenerator(paths, labels, )
   testg = ProteinDataGenerator(pathsTest, labelsTest)
   cfg=classification.parse("./densenet201-flips/proteins.yaml")
   cfg.setAllowResume(True)
   cfg.fit(tg,foldsToExecute=[0],start_from_stage=1)
   lastFullValPred ,lastFullValLabels =cfg.evaluate_all_to_arrays(tg,0,0,ttflips=True)

   TP,mn=getOptimalT(lastFullValPred,lastFullValLabels)
   submit = pd.read_csv(DIR + '/sample_submission.csv')
   prediction = []
   vd = cfg.validation(tg,0)
   P=cfg.predict_all_to_array(testg, 0, 0, ttflips=True)

   for row in tqdm(range(submit.shape[0])):
         str_label = ''
         for col in range(P.shape[1]):
             if (P[row, col] < TP[col]):
                 str_label += ''
             else:
                 str_label += str(col) + ' '
         prediction.append(str_label.strip())
   submit['Predicted'] = np.array(prediction)
   submit.to_csv('4channels_cnn_from_scratch.csv', index=False)
if __name__ == '__main__':
    main()