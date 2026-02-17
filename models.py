import numpy as np
import matplotlib.pyplot as plt

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
    return prediction - target 
  
  def fit(self, X, y, tol=1e-5, epoch=1000):
    self.history = []
    n = len(X)

    for _ in range(epoch):
        y_pred = self.predict(X)
        error = y_pred - y
        
        mse = np.mean(error ** 2)
        self.history.append(mse)

        if mse < tol:
            break

        dw = (2/n) * np.dot(error, X)
        db = (2/n) * np.sum(error)

        self.weights[1:] -= self.learning_rate * dw
        self.weights[0] -= self.learning_rate * db


if __name__ == "__main__":
   k, b = 1, 2

   x = np.linspace(-10, 11, 1000)
   x = x.reshape(-1, 1)

   y = k*x + b
   y = y.flatten()

   random_int = (np.random.rand(1000) - 0.5) * 10
   y_synt = y + random_int

   nn = Perceptron(1, learning_rate=0.001)
   nn.fit(x, y_synt)


   plt.plot(x, y_synt, 'o', c = 'r')
   plt.plot(x, y)
   plt.show()
   print(y)

   nn = Perceptron(1)
   nn.fit(x, y)
   print(nn.predict(x))
   print("Weights:", nn.weights)