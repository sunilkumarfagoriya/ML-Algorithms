import numpy as np

def euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn(X_train, y_train, X_test, k):
    y_final = list()

    for x in X_test:
        y_pred = list()
        
        i = 0
        for y in X_train:
            y_pred.append([euclidean(x, y), i])
            i += 1
        
        y_pred.sort()

        k_pred = list()
        for y in range(k):
            k_pred.append(y_pred[y])
        
        y_pred = list()
        for y in range(k):
            y_pred.append(y_train[k_pred[y][1]])

        y_final.append(max(set(y_pred), key = y_pred.count))

    return y_final



X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [2, 2],
    [7, 7]
])

k = 3

print(knn(X_train, y_train, X_test, k))