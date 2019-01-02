import sklearn.model_selection as ms
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
v=ms.StratifiedShuffleSplit(n_splits=5,test_size=0.4,random_state=0)

for train_index, test_index in v.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("A")