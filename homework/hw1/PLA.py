import numpy as np
from random import randrange
from sklearn import utils

def PLAClassify(X, W):
    X1 = np.insert(X, X.shape[1], values=1, axis=1)
    return np.sign(np.dot(X1, W))


def PLATrain(X, Y, rand=False, rate=1, times=100, pocket=True):
    W = np.zeros(X.shape[1]+1)
    idx = -1
    count = 0
    if rand:
        X, Y = utils.shuffle(X, Y)
    if pocket:
        rand = True
        W1 = np.zeros(X.shape[1]+1)
        last_error = X.shape[0]

    while(count <= times):
        error = np.where(Y != PLAClassify(X, W))[0]
        if error.shape[0] != 0:
            rest = error[error > idx]
            if rest.shape[0] != 0:
                idx = error[error > idx][0]
            else:
                idx = -1
                continue
            if pocket:
                W1 = W + np.append(X[idx, :],1)*Y[idx]*rate
                now_error = sum(Y != PLAClassify(X, W1))
                if now_error < last_error:
                    W = W1
                    last_error = now_error
            else:
                W += np.append(X[idx, :],1)*Y[idx]*rate
        else:
            return W, count
        count += 1

    return W, count
