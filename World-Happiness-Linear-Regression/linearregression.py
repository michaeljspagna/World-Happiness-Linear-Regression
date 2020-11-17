import numpy as np

class LinearRegressionModel(object):
    """
    This class performs Linear regression on a set of input data
    """
    def __init__(self, featureCount) -> None:
        self.featureCount = featureCount
        self.thetas = np.random.randn(featureCount,1)
        
    def _hypothesis(self, X):
        """
        X: (m x n) matrix
        thetas: (n x 1) vector
        hypothesis(X) = X dot (theta's)
        """
        return np.dot(X, self.thetas)
    
    def _computeCost(self, y, prediction):
        """
        y: (m x 1) vector of true y values
        prediction: (m x 1) vector of predicted y values
        cost = (1/2m) * sum((prediction - y)^2)
        """
        m = y.shape[0]
        cost = np.sum(np.square(prediction - y))
        return cost / (2 * m)
    
    def _calculateNewThetas(self, X, y, prediction):
        """
        X: (m x n) matrix of feature data
        y: (m x 1) vector of true y values
        prediction: (m x 1) vector of predicted y values
        dTheta = sum((prediction-y)^T dot X) / m
        """
        m = y.shape[0]
        dTheta = np.sum(np.dot(np.transpose(prediction - y), X), axis=0) / m
        return dTheta
    
    def _updateThetas(self, dTheta, alpha):
        """
        dTheta: (n x 1) vector of updated theta values
        alpha: scalar learning rate to scale down value to update thetas
        newThetas = oldThetas - alpha(dTheta)
        """
        self.thetas = self.thetas - alpha * np.reshape(dTheta, (self.featureCount,1))
        
    def train(self, X_train, Y_train, alpha, iterations, display=1):
        """
        X_train: (m x n) matrix of feature data
        Y_train: (m x 1) vector of true y values
        alpha: scalar learning rate
        iterations: scalar number of iterations Gradient Decent should run
        repeat until converge:{
            1. Make a prediction
            2. Calculate new theta values
            3. Update current theta values
            4. Compute and save cost
                - Cost should decrease with every iteration
        }
        """
        costs = []
        cost = 1
        i = 0
        for i in range(iterations):
            prediction = self._hypothesis(X_train)
            dTheta = self._calculateNewThetas(X_train, Y_train, prediction)
            self._updateThetas(dTheta, alpha)
            cost = self._computeCost(Y_train, prediction)
            costs.append(cost)
            if display:
                if i % 100 == 0:
                    print('Iter: {}, Current loss: {:.4f}'.format(i, cost))
            i += 1
        return costs
            
    def test(self, X_test, Y_test, display=1):
        """
        X_test: (m x n) matrix of feature values
        Y_test: (m x 1) vector of true y values
        To test our Model{
            1. Make a prediction based off input X_test[i]
            2. Compare to corresponding input Y_test[i]
            3. Compute cost
                -Cost should be low
        }
        """
        prediction = np.apply_along_axis(self._hypothesis, axis=1, arr=X_test)
        for i in range(Y_test.shape[0]):
            cost = self._computeCost(Y_test[i], prediction[i])
            if display: 
                print('Predicted: {:.4f} Actual: {:.4f} Cost: {:.4f}'.format(prediction[i,0], Y_test[i,0], cost))
                