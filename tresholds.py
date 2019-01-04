import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score as off1

def getOptimalT(lastFullValPred, lastFullValLabels):
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
    print('Choosing threshold: ', T, ', validation F1-score: ', np.mean(np.max(f1s, axis=0)))
    print(T)
    return T, np.mean(np.max(f1s, axis=0))


def getOptimalT2(lastFullValPred, lastFullValPred1, lastFullValLabels):
    rng = np.arange(0, 1, 0.001)
    rng1 = np.arange(0, 1, 0.05)
    f1s = np.zeros((rng1.shape[0], rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for j1, t1 in enumerate(tqdm(rng1)):
            for i in range(28):
                p = np.array(lastFullValPred[:, i] * (1 - t1) + lastFullValPred1[:, i] * (t1) > t, dtype=np.int8)
                scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
                f1s[j1, j, i] = scoref1

    print(np.max(np.max(f1s, axis=1), axis=0))
    print(np.mean(np.max(np.max(f1s, axis=1), axis=0)))

    T = np.empty(28)
    T1 = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, :, i] == np.max(f1s[:, :, i]))[1][0]]
        T1[i] = rng1[np.where(f1s[:, :, i] == np.max(f1s[:, :, i]))[0][0]]
    print('Choosing threshold: ', T, ', validation F1-score: ', np.mean(np.max(f1s, axis=0)))
    print(T)
    return T, T1, np.mean(np.max(f1s, axis=0))