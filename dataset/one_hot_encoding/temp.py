import numpy as np

y_train = np.loadtxt("y_train.csv", delimiter=",", dtype=int)
y_cv = np.loadtxt("y_cv.csv", delimiter=",", dtype=int)
y_test = np.loadtxt("y_test.csv", delimiter=",", dtype=int)

y_train = np.unique(y_train, return_counts=True)
y_train = dict(zip(y_train[0], y_train[1]))
y_cv = np.unique(y_cv, return_counts=True)
y_cv = dict(zip(y_cv[0], y_cv[1]))
y_test = np.unique(y_test, return_counts=True)
y_test = dict(zip(y_test[0], y_test[1]))

total = 0

for i in y_train:
    total += y_train[i] + y_cv[i] + y_test[i]
    y_train[i] += y_cv[i] + y_test[i]

print(y_train[0]/total, y_train[1]/total, y_train[2]/total)

for i in range(1,4):
    y_train = np.loadtxt(f"y_train_subset_{i}.csv", delimiter=",", dtype=int)
    total = len(y_train)
    y_train = np.unique(y_train, return_counts=True)
    y_train = dict(zip(y_train[0], y_train[1]))
    print(i, y_train[0]/total, y_train[1]/total, y_train[2]/total)
