from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pplt

def evaluate(yreal, prediction):
    msqe = mean_squared_error(yreal, prediction)
    print('Mean Square Error: ' + msqe)
    pplt.plot(yreal)
    pplt.plot(prediction)
    pplt.show()