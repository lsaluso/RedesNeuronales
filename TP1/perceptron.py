import numpy as np


class Perceptron:

    def __init__(self, nu=0.01):
        self._w0 = np.random.uniform(low=-0.5, high=0.5)
        self._w1 = np.random.uniform(low=-0.5, high=0.5)
        self._w2 = np.random.uniform(low=-0.5, high=0.5)
        self._x0 = -1
        self._nu = nu
        self._y = 0

    def calculate(self, row):
        self._y = np.sign(self._w0 * self._x0 + self._w1 * row['x1'] + self._w2 * row['x2'])
        self._w0 = self._w0 + self._nu * (row['yd'] - self._y) * self._x0
        self._w1 = self._w1 + self._nu * (row['yd'] - self._y) * row['x1']
        self._w2 = self._w2 + self._nu * (row['yd'] - self._y) * row['x2']

    def get_w0(self):
        return self._w0

    def get_w1(self):
        return self._w1

    def get_w2(self):
        return self._w2

    def get_y(self):
        return self._y
    
    def __str__(self):
        cad  = "w0: " + str(self._w0) + "\n"
        cad += "w1: " + str(self._w1) + "\n"
        cad += "w2: " + str(self._w2) + "\n"
        cad += "y: " + str(self._y) + "\n"
        return cad