import numpy as np
from sklearn.metrics.classification import f1_score

class DataGenerator:

    def __init__(self,fraction,noiseProb,count):
        self.fraction=fraction
        self.noiseProb=noiseProb
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

def noise(input,mean,variance):
    m= input+np.random.normal(mean,variance,input.shape[0])
    return (m-m.min())/(m.max()-m.min())

def estimate(d,v,tr):
    return f1_score(d > 0.5, v>tr)

dg=DataGenerator(0.1, 0.2,1000);

v=dg.get()

def toPred(v,num):
    d=None
    for i in range(num):
        if d is None:
            d=noise(v,0.5,0.6)
        else d=d+noise(v,0.5,0.6)
    d=d/len(num)
    return d;

def optimal(v,d):
    maxVal=0;
    t=0;
    for i in range(20):
        val=estimate(v,d,i*0.05)
        if val>maxVal:
            maxVal=val
            t=i*0.05
    return maxVal,t


r=optimal(v,toPred(v))

test=toPred(dg.validation())

print(r)

print(estimate(dg.validation(),test,r[1]))