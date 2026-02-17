import numpy as np

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.01):
    self.weights = np.random.randn(num_inputs + 1)
    self.learning_rate = learning_rate

  def weighted_sum(self, inputs):
    Z=np.dot(inputs, self.weights[1:]) + self.weights[0]
    return Z
  
  def predict(self, X):
    z=self.weighted_sum(X)
    return z
  
  def loss(self, prediction, target):
    return target - prediction
  
  def fit(self, X, y, tol=10e-5, epoch=100):
    self.history = []
    for _ in range(epoch):
        y_pred = self.predict(X)
        mse = self.loss(y_pred, y)**2
        if mse<tol:
            pass 
        else:
            self.weights[1:] = self.weights[1:]-2*X*mse**0.5
            self.weights[0] = self.weights[0]-2*mse**0.5 
            self.history.append(mse.mean())