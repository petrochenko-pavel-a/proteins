import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import os

DIR="D:/cells/";

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


def split(X,y):

    all=np.arange(y.shape[0])


    currentSet=set(all)

    rs=[]
    for c in range(y.shape[1]):
        goodClasss=np.where(y[:,c]>0)[0]
        rs.append(len(goodClasss))
        print(len(goodClasss),c)
        len(goodClasss)
    zz=np.argsort(np.array(rs))

    trainSet=set()
    testSet=set()

    np.random.seed(12)
    if os.path.exists("train.txt"):
        with open("train.txt","r") as f:
            trainLines=[x.strip() for x in f.readlines()]
        with open("test.txt","r") as f:
            testLines=[x.strip() for x in f.readlines()]
            #just for debug
        for v in range(y.shape[0]):
            if X[v] in trainLines:
                trainSet.add(v)
            else:
                testSet.add(v)
    else:
        for v in zz:
            # now we should start choosing examples
            goodClasss = np.where(y[:, v] > 0)[0]
            np.random.shuffle(goodClasss)
            test=0
            train=0
            for c in goodClasss:

                if test*5<train:
                    if not c in trainSet:
                        testSet.add(c)
                        test=test+1
                    else:
                        train=train+1
                        trainSet.add(c);
                else:
                    if not c in testSet:
                        train = train + 1
                        trainSet.add(c);
                    else:
                        testSet.add(c)
                        test = test + 1
            print(test,train)
        # vv=goodClasss
        # vv1=vv[:vv.shape[0]//5]
        # vv2=vv[vv.shape[0]//5:]
    trainX,trainY=X[[np.array(list(trainSet))]], y[np.array(list(trainSet)),:]
    testX, testY = X[[np.array(list(testSet))]], y[np.array(list(testSet)), :]
    return ([trainX,trainY],[testX,testY])