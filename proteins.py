import numpy as np
from musket_core.datasets import PredictionItem
import musket_core.datasets as ds
import pickle
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

DIR = 'D:/cells'

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


def getOptimalT(lastFullValPred,lastFullValLabels):
    rng = np.arange(0, 1.4, 0.001)
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

def getOptimalT2(lastFullValPred,lastFullValPred1,lastFullValLabels):
    rng = np.arange(0, 1, 0.001)
    rng1 = np.arange(0, 1, 0.05)
    f1s = np.zeros((rng1.shape[0],rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for j1, t1 in enumerate(tqdm(rng1)):
            for i in range(28):
                p = np.array(lastFullValPred[:, i]*(1-t1)+lastFullValPred1[:, i]*(t1) > t, dtype=np.int8)
                scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
                f1s[j1, j, i] = scoref1

    print(np.max(np.max(f1s,axis=1),axis=0))
    print(np.mean(np.max(np.max(f1s,axis=1),axis=0)))

    T  = np.empty(28)
    T1 = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, :,i] == np.max(f1s[:, : , i]))[1][0]]
        T1[i] = rng1[np.where(f1s[:, :, i] == np.max(f1s[:, :, i]))[0][0]]
    print('Choosing threshold: ', T, ', validation F1-score: ', np.mean(np.max(f1s,axis=0)))
    print(T)
    return T,T1, np.mean(np.max(f1s, axis=0))



def save(p,d):
    with open(p,"wb") as f:
        pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def main():
   paths, labels = getTrainDataset()
   tg = ProteinDataGenerator(paths, labels)

   pathsTest, labelsTest = getTestDataset()
   testg = ProteinDataGenerator(pathsTest, labelsTest)

   paths2, labels2 = getTrainDataset2()

   tg2 = ProteinDataGenerator(paths2, labels2)


   cfg=classification.parse("./xc2/proteins.yaml")
   cfg0=classification.parse("./xception-clr/proteins.yaml")
   #cfg2 = classification.parse("./xc2/proteins.yaml")
   classification.extra_train["train2"]=tg2
   # finder=cfg.lr_find(tg,stage=2,epochs=1)
   # finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
   # plt.show()
   # finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
   # plt.show()

   #cfg0=classification.parse("C:/Users/Павел/PycharmProjects/classification_training_pipeline/examples/proteins2/proteins.yaml")
   cfg.gpus=2
   cfg.setAllowResume(True)
   #ds.USE_MULTIPROCESSING=True
   #cfg.fit(tg,foldsToExecute=[3],start_from_stage=1)



   # lastFullValPred1, lastFullValLabels = cfg0.evaluate_all_to_arrays(tg, 3, 2, ttflips=True,batchSize=128)
   # lastFullValPred, lastFullValLabels = cfg.evaluate_all_to_arrays(tg, 3, 0, ttflips=True,batchSize=128)
   # lastFullValPred2, lastFullValLabels = cfg0.evaluate_all_to_arrays(tg, 3, 1, ttflips=True,batchSize=128)
   # lastFullValPred3, lastFullValLabels = cfg.evaluate_all_to_arrays(tg, 3, 1, ttflips=True,batchSize=128)
   # TPX, mn = getOptimalT(lastFullValPred+lastFullValPred1+lastFullValPred2,lastFullValLabels)
   #
   #
   #save("BDE",TPX)

   #FOLD_1_TRESH, mn = getOptimalT(v2,l)
   FOLD_1_MIX_TRESH = load("BBB")
   FOLD_0_MIX_TRESH=load("BA")
   FOLD_1_TRESH=load("AB")
   FOLD_2_MIX_TRESH = load("BCC")
   FOLD_3_MIX_TRESH = load("BDE")
   FOLD_3_TRESH= load("AD")
   FOLD_4_TRESH=load("AE")
   FOLD_1_1_TRESH=load("AF")
   #save("AF",FOLD_1_1_TRESH)
   #save("AB", FOLD_1_TRESH)
   #FOLD_0_MIX_TRESH=(FOLD_0_MIX_TRESH+FOLD_1_TRESH)/2

   submit = pd.read_csv(DIR + '/sample_submission.csv')
   prediction = []

   # P5=cfg.predict_all_to_array(testg, 3, 1, ttflips=True,batch_size=64)
   # #
   # save("./store/3xfa512.pred_dat",P5)
   # #
   # P5 = cfg0.predict_all_to_array(testg, 3, 2, ttflips=True, batch_size=64)
   # save("./store/3xfa256.pred_dat", P5)
   # #
   #exit(0)

   FOLD_2_1=load("./store/2xc512.pred_dat")
   FOLD_2_2 = load("./store/2xf512.pred_dat")
   FOLD_4=load("./store/4xc256.pred_dat")
   FOLD_0_2=load("./store/0cx512.pred_dat")
   FOLD_0_0=load("./store/0nn256.pred_dat")
   FOLD_0_1 = load("./store/0xc256.pred_dat")
   FOLD_2 = load("./store/2xc256.pred_dat")
   FOLD_1 =load("./store/1xc256.pred_dat")

   FOLD_0_JOINT=(FOLD_0_0+FOLD_0_1+FOLD_0_2)
   FOLD_2_JOINT = (FOLD_2_1 + FOLD_2+FOLD_2_2)

   FOLD_1_1 = load("./store/1xc512.pred_dat")
   FOLD_1_2 = load("./store/1xfa512.pred_dat")
   FOLD_1_3 = load("./store/1xfa256.pred_dat")

   FOLD_1_JOINT=FOLD_1+FOLD_1_1+FOLD_1_2+FOLD_1_3;

   FOLD_3=load("./store/3xc256.pred_dat")
   FOLD_3_1 = load("./store/3xc512.pred_dat")
   FOLD_3_2 = load("./store/3xfa512.pred_dat")
   FOLD_3_3 = load("./store/3xfa256.pred_dat")
   FOLD_3_JOINT=FOLD_3+FOLD_3_1+FOLD_3_2+FOLD_3_3

   FOLD_GUYS = load("./store/resnet34_repro_v16_with_external_TTAx16_lb0.567_with_leak.h5.pred_data.max.sorted")
   TRESH_GUYS= np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])

   CP=[
       (FOLD_0_JOINT,FOLD_0_MIX_TRESH),
       (FOLD_1,FOLD_1_TRESH),
       (FOLD_2_MIX_TRESH, FOLD_2_JOINT),
       (FOLD_3, FOLD_3_TRESH),
       (FOLD_4, FOLD_4_TRESH)
       ]

   OTHER_FOLDS=FOLD_4+FOLD_GUYS
   OTHER_FOLDS_TRESH=FOLD_4_TRESH+TRESH_GUYS;
   #3+2+2+2+2
   #6
   SUM_TRESH= FOLD_0_MIX_TRESH*0.7 + FOLD_1_MIX_TRESH * 0.7 + FOLD_2_MIX_TRESH * 0.7+ OTHER_FOLDS_TRESH*0.7+FOLD_3_MIX_TRESH*0.7


   SUM_FOLD=FOLD_0_JOINT+ FOLD_1_JOINT +FOLD_2_JOINT + FOLD_3_JOINT+OTHER_FOLDS
   save("all.dat",SUM_FOLD)
   save("allTresh.dat", SUM_TRESH)
   for row in tqdm(range(submit.shape[0])):
         str_label = ''
         for col in range(FOLD_0_0.shape[1]):
             if SUM_FOLD[row, col]< SUM_TRESH[col]:
                 str_label += ''
             else:
                 str_label += str(col) + ' '
         cm=0.95
         while len(str_label.strip())==0:
             str_label = ''
             for col in range(FOLD_0_0.shape[1]):
                 if SUM_FOLD[row, col] < SUM_TRESH[col]*cm:
                     str_label += ''
                 else:
                     str_label += str(col) + ' '
             cm = cm*0.95

         prediction.append(str_label.strip())
   submit['Predicted'] = np.array(prediction)
   submit.to_csv('4channels_cnn_from_scratch.csv', index=False)
if __name__ == '__main__':
    main()
