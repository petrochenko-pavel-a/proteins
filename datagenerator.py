import numpy as np
from sklearn.metrics.classification import f1_score

class DataGenerator:

    def __init__(self,fraction,count):
        self.fraction=fraction
        self.data=None
        self.val = None
        self.count=count;
        pass

    def get(self):
        if self.data is not None:
            return self.data
        zeros=np.zeros(self.count)
        zeros[:round((self.count*self.fraction))]=1;
        np.random.shuffle(zeros)
        self.data=zeros;
        return zeros

    def validation(self):
        if self.val is not None:
            return self.val
        zeros=np.zeros(self.count)
        zeros[:round((self.count*self.fraction))]=1;
        np.random.shuffle(zeros)
        self.val=zeros;
        return zeros

class Noised:
    def __init__(self,mean,variance,len):
        self.noise=np.random.normal(mean,variance,len)

    def apply(self,input):
        m = input + self.noise;
        return (m - m.min()) / (m.max() - m.min())


def estimate(d,v,tr):
    return f1_score(d > 0.5, v>tr)

dg=DataGenerator(0.05, 1000);

def optimal(v,d):
    maxVal=0;
    t=0;
    for i in range(20):
        val=estimate(v,d,i*0.05)
        if val>maxVal:
            maxVal=val
            t=i*0.05
    return maxVal,t

v=dg.get()

import random

def est(count):
    est=[]
    for i in range(0,100):
        estimators = []
        rates = []
        for i in range(count):
            n=Noised(random.random(),0.6+random.random(),1000)
            estimators.append(n)
            rates.append(optimal(v,n.apply(v)))

        #print(optimal(v,n1.apply(v)))

        test=dg.validation()

        allPred=[]
        sr=0
        for i in range(len(estimators)):
            n=estimators[i]
            #tr=rates[i][1]
            sr=sr+rates[i][1]
            predictions=n.apply(test);
            allPred.append(predictions);
            #print(estimate(test,predictions,tr))

        allPred=np.array(allPred).sum(axis=0)
        est.append(estimate(test, allPred, sr))
    return np.array(est).mean()

for i in range(2,15):
    print(i,est(i))


#print(allPred)