from copy import deepcopy

import numpy as np


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.elite = None
        self.evaluations = 0

    def Evaluate(self, individual):
        self.evaluations = self.evaluations + 1

        output = individual.GetOutput(self.X_train)
        print(self.y_train.shape, output.shape, individual.GetSubtree())
        b = np.cov(self.y_train, output)[0, 1] / np.var(output)
        a = np.mean(self.y_train) - b * np.mean(output)

        mean_squared_error = np.mean(np.square(self.y_train - (a + b * output)))
        individual.fitness = mean_squared_error

        if not self.elite or individual.fitness < self.elite.fitness:
            del self.elite
            self.elite = deepcopy(individual)
