import bulbea as bb
from bulbea.learn.evaluation import split
import numpy as np
import quandl

quandl.ApiConfig.api_key = "ppsopkmQbca4mgpMP_aa"
share = bb.Share('WIKI', 'GOOGL')
share.data

Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)


Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
Xtest  = np.reshape(Xtest, (Xtest.shape[0],  Xtest.shape[1], 1))

def get_data():
    return (Xtrain, ytrain), (Xtest, ytest)
