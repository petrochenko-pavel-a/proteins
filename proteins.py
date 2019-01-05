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
import split_data
import loaders


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
   paths, labels = loaders.getTrainDataset()

   trainD,testD=split_data.split(paths, labels)


   tg = loaders.ProteinDataGenerator(trainD[0], trainD[1] )

   with open("train.txt","w") as f:
       for l in trainD[0]:
           f.write(l+"\r")
   with open("test.txt","w") as f:
       for l in testD[0]:
           f.write(l+"\r")

   paths2, labels2 = loaders.getTrainDataset2()

   extra_data = loaders.ProteinDataGenerator(paths2, labels2)
   classification.extra_train["train2"] = extra_data

   cfg=classification.parse("./densenet201/proteins.yaml")
   cfg0=classification.parse("./xception-clr/proteins.yaml")

   # finder=cfg.lr_find(tg,stage=2,epochs=1)
   # finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
   # plt.show()
   # finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
   # plt.show()
   #exit(0)
   #cfg0=classification.parse("C:/Users/Павел/PycharmProjects/classification_training_pipeline/examples/proteins2/proteins.yaml")
   cfg.gpus=2
   cfg.setAllowResume(True)
   #ds.USE_MULTIPROCESSING=True
   cfg.fit(tg,foldsToExecute=[0],start_from_stage=0)
   #lastFullValPred ,lastFullValLabels =load("./store/0xc256.eval_dat")
   #lastFullValPred1, lastFullValLabels1 = load("./store/0nn256.eval_dat")
   #TP1,FOLD_0_MIX_TRESH,m=getOptimalT2(lastFullValPred,lastFullValPred1,lastFullValLabels)
   #save("./store/mix00.eval",(TP1,FOLD_0_MIX_TRESH))

   #FOLD_0_MIX_TRESH, TP1 =load("./store/mix00.eval")
   #exit(0)

   #lastFullValPred, lastFullValLabels = cfg.evaluate_all_to_arrays(tg,1,0,ttflips=True)

   #save("./store/0xc256.eval_dat",(lastFullValPred ,lastFullValLabels))
   #exit(0)
   #lastFullValPred1, lastFullValLabels1 = cfg0.evaluate_all_to_arrays(tg, 1, 0, ttflips=True)
   # lastFullValPred2, lastFullValLabels2 = cfg.evaluate_all_to_arrays(tg, 0, 0, ttflips=True)

   # lastFullValPred=(lastFullValPred2*0.5+lastFullValPred1*0.5)/2

   v,l=load("./store/0nn256.eval_dat")
   v1,l=load("./store/0xc256.eval_dat")
   #v1=lastFullValPred1
   v2,l=load("./store/1xcf256.eval_dat")
   FOLD_1_MIX_TRESH=load("BB")
   #save("BB",FOLD_1_1_TRESH)

   #TP1, mn = getOptimalT(lastFullValPred2,lastFullValLabels2)

   #FOLD_1_TRESH, mn = getOptimalT(v2,l)

   FOLD_0_MIX_TRESH=load("AA")
   FOLD_1_TRESH=load("AB")
   FOLD_2_TRESH = load("AC")
   FOLD_3_TRESH= load("AD")
   FOLD_4_TRESH=load("AE")
   FOLD_1_1_TRESH=load("AF")
   #save("AF",FOLD_1_1_TRESH)
   #save("AB", FOLD_1_TRESH)
   #FOLD_0_MIX_TRESH=(FOLD_0_MIX_TRESH+FOLD_1_TRESH)/2

   submit = pd.read_csv(DIR + '/sample_submission.csv')
   prediction = []
   vd = cfg.validation(tg,0)
   #P5=cfg.predict_all_to_array(testg, 1, 0, ttflips=True,batch_size=64)
   P5=load("./store/1cx256.pred_dat")
   #FOLD_1 = cfg.predict_all_to_array(testg, 1, 0, ttflips=True, batch_size=64)
   FOLD_4=load("./store/4xc256.pred_dat")

   FOLD_0_0=load("./store/0nn256.pred_dat")
   FOLD_0_1 = load("./store/0xc256.pred_dat")
   FOLD_2 = load("./store/2xc256.pred_dat")
   FOLD_1 =load("./store/1xc256.pred_dat")
   FOLD_0_JOINT=(FOLD_0_0+FOLD_0_1)/2
   #FOLD_1=(FOLD_1+(FOLD_0_0+FOLD_0_1)/2)/3

   #print(FOLD_1)
   FOLD_1_1 = load("./store/1xc512.pred_dat")
   FOLD_3=load("./store/3xc256.pred_dat")
   #FOLD_1=(FOLD_1*0.5+FOLD_3*0.5)/2
   #FOLD_0_JOINT[row, col]*0.55+FOLD_1[row,col]*0.45+FOLD_2[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.55+FOLD_1_TRESH[col]*0.45+FOLD_2_TRESH[col]*0.3
   #FOLD_0_JOINT[row, col]*0.6+FOLD_1[row,col]*0.45+FOLD_2[row,col]*0.2+FOLD_3[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.55+FOLD_1_TRESH[col]*0.3+FOLD_2_TRESH[col]*0.2+FOLD_3_TRESH[col]*0.3
   #FOLD_0_JOINT[row, col]*0.5+FOLD_1[row,col]*0.4+FOLD_2[row,col]*0.15+FOLD_3[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.5+FOLD_1_TRESH[col]*0.4+FOLD_2_TRESH[col]*0.15+FOLD_3_TRESH[col]*0.3

   CP=[
       (FOLD_0_JOINT,FOLD_0_MIX_TRESH),
       (FOLD_1,FOLD_1_TRESH),
       (FOLD_2, FOLD_2_TRESH),
       (FOLD_3, FOLD_3_TRESH),
       (FOLD_4, FOLD_4_TRESH)
       ]

   for row in tqdm(range(submit.shape[0])):
         str_label = ''
         for col in range(FOLD_0_0.shape[1]):

             if FOLD_0_JOINT[row,col]*0.7+FOLD_1_1[row,col]+FOLD_1[row,col]+FOLD_2[row,col]*0.2+FOLD_4[row,col]*0.2+FOLD_3[row,col]*0.2 <\
                     FOLD_1_MIX_TRESH[col]+FOLD_0_MIX_TRESH[col]*0.7+FOLD_2_TRESH[col]*0.2+FOLD_4_TRESH[col]*0.2+FOLD_3_TRESH[col]*0.2:
                 str_label += ''
             else:
                 str_label += str(col) + ' '
         prediction.append(str_label.strip())
   submit['Predicted'] = np.array(prediction)
   submit.to_csv('4channels_cnn_from_scratch.csv', index=False)
if __name__ == '__main__':
    main()