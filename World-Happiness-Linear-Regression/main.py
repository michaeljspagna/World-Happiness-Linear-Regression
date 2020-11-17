from datahandler import DataHandler
from linearregression import LinearRegressionModel


if __name__ == "__main__":
    dh = DataHandler()
    X_train, Y_train, X_test, Y_test = dh.getTrainAndTest()
    modelT = LinearRegressionModel(X_train.shape[1])
    modelT.train(X_train, Y_train, 0.5, 700)
    modelT.test(X_test, Y_test)