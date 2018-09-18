from bulbea.learn.models import RNN

rnn = RNN([1, 100, 100, 1])

def fit(X, y):
    rnn.fit(X, y)

def predict(X):
    return rnn.predict(X)