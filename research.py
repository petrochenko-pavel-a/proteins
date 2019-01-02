import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)
FOLD_0_MIX_TRESH=load("AA")
FOLD_1_TRESH=load("AB")
FOLD_2_TRESH = load("AC")
FOLD_3_TRESH= load("AD")
FOLD_4_TRESH=load("AE")
FOLD_1_1_TRESH=load("AF")
#save("AF",FOLD_1_1_TRESH)
#save("AB", FOLD_1_TRESH)
#FOLD_0_MIX_TRESH=(FOLD_0_MIX_TRESH+FOLD_1_TRESH)/2

prediction = []
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
#FOLD_3 = cfg.predict_all_to_array(testg, 3, 1, ttflips=True,batch_size=64)

FOLD_3=load("./store/3xc256.pred_dat")
#FOLD_1=(FOLD_1*0.5+FOLD_3*0.5)/2
#FOLD_0_JOINT[row, col]*0.55+FOLD_1[row,col]*0.45+FOLD_2[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.55+FOLD_1_TRESH[col]*0.45+FOLD_2_TRESH[col]*0.3
#FOLD_0_JOINT[row, col]*0.6+FOLD_1[row,col]*0.45+FOLD_2[row,col]*0.2+FOLD_3[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.55+FOLD_1_TRESH[col]*0.3+FOLD_2_TRESH[col]*0.2+FOLD_3_TRESH[col]*0.3
#FOLD_0_JOINT[row, col]*0.5+FOLD_1[row,col]*0.4+FOLD_2[row,col]*0.15+FOLD_3[row,col]*0.3 < FOLD_0_MIX_TRESH[col]*0.5+FOLD_1_TRESH[col]*0.4+FOLD_2_TRESH[col]*0.15+FOLD_3_TRESH[col]*0.3

DIR = 'D:/cells'
CP=[
   (FOLD_0_JOINT,FOLD_0_MIX_TRESH),
   (FOLD_1,FOLD_1_TRESH),
   (FOLD_2, FOLD_2_TRESH),
   (FOLD_3, FOLD_3_TRESH),
   (FOLD_4, FOLD_4_TRESH)
   ]

submit = pd.read_csv(DIR + '/sample_submission.csv')
R=None
for x,y in CP:

    #print((x-y).mean(axis=0))
    #print(((x - y)> 0).sum(axis=0))
    #print(((x - y)/y > 0).sum(axis=0))

    r=np.clip((x - y)/y ,-1,30)
    if R is None:
        R=r
    else:
        R=R+r
    gt=(R>0).sum(axis=0)
    print(gt)
    print(gt.sum())

count=0
for row in tqdm(range(submit.shape[0])):
         str_label = ''
         for col in range(FOLD_0_0.shape[1]):

             if FOLD_0_JOINT[row, col]*0.55+FOLD_1[row,col]*0.45+FOLD_2[row,col]*0.15+FOLD_3[row,col]*0.3+FOLD_4[row,col]*0.005 < FOLD_0_MIX_TRESH[col]*0.55+FOLD_1_TRESH[col]*0.45+FOLD_2_TRESH[col]*0.15+FOLD_3_TRESH[col]*0.25+FOLD_4_TRESH[col]*0.03:
                 str_label += ''
             else:
                 str_label += str(col) + ' '
                 count = count + 1
         prediction.append(str_label.strip())

submit['Predicted'] = np.array(prediction)

submit.to_csv('4channels_cnn_from_scratch.csv', index=False)
print(count)