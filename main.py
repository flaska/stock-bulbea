import data
import model
import eval

(Xtrain, ytrain), (Xtest, ytest) = data.get_data()
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)

eval.evaluate(ytest, prediction)

