import numpy as np
from random import randrange


def PLAClassify(X, W):
    X1 = np.insert(X, X.shape[1], values=1, axis=1)
    return np.sign(np.dot(X1, W))


def PLATrain(X, Y, rand=False, rate=1, times=1000, pocket=True):
    W = np.zeros(X.shape[1]+1)
    x1 = np.insert(X, X.shape[1], values=1, axis=1)
    if pocket:
        rand = True
        W1 = np.zeros(X.shape[1]+1)
        last_error = X.shape[0]

    for count in range(times):
        error = np.where(Y != PLAClassify(X, W))[0]
        if error.shape[0] != 0:
            idx = error[randrange(0, error.shape[0]) if rand else 0]
            if pocket:
                W1 = W + x1[idx, :]*Y[idx]*rate
                now_error = sum(Y != PLAClassify(X, W1))
                if now_error < last_error:
                    W = W1
                    last_error = now_error
            else:
                W += x1[idx, :]*Y[idx]*rate
        else:
            return W, count
    return W, count
