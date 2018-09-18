import data
from bulbea.learn.models import RNN

(Xtrain, Xtest), (ytrain, ytest) = data.get_data()

rnn = RNN([1, 100, 100, 1])
rnn.fit(Xtrain, ytrain)

from sklearn.metrics import mean_squared_error
p = rnn.predict(Xtest)
mean_squared_error(ytest, p)
import matplotlib.pyplot as pplt
pplt.plot(ytest)
pplt.plot(p)
pplt.show()