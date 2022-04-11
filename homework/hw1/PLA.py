import numpy as np
from random import randrange


def PLAClassify(X, W):
    X1 = np.insert(X, X.shape[1], values=1, axis=1)
    return np.sign(X1.dot(W))


def PLATrain(X, Y, rand=False, rate=1, times=1000, pocket=True):
    W = np.zeros(X.shape[1]+1)
    last_error = 0
    if pocket:
        rand = True
    for count in range(times):
        error = np.where(Y != PLAClassify(X, W))[0]
        if len(error) != 0:
            idx = error[randrange(0, len(error)) if rand else 0]
            W1 = W+np.append(X[idx, :], 1)*Y[idx]*rate
            now_error = sum(Y != PLAClassify(X, W1))
            if pocket and now_error > last_error:
                pass
            else:
                W = W1
                last_error = now_error
        else:
            return W, count
    return W, count
