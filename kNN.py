import numpy as np

class kNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def calculate_accurracy(self, predicts, y):
        f = np.frompyfunc(lambda x: x if x == 1 else 0,1,1)
        num_correct_predicts = np.sum(f(predicts*y))
        num_predicts = y.shape[1]
        acc = num_correct_predicts / num_predicts
        return acc
    
    def euclidean_distance(self, x, X):
        return np.sum((x - X)**2, axis=0)

    def voting(self, k_nearest, distances, labels):
        scores = np.zeros((2,1))
        for neighbour in k_nearest:
            d = distances[neighbour]
            if d == 0:
                return labels[:,neighbour]

            if labels[:,neighbour] == 1:
                scores[0] += 1/d
            else:
                scores[1] += 1/d

        return 1 if scores[0] > scores[1] else -1

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test, y_test=None, k=10):
        predicts = np.zeros((1, y_test.shape[1]))
        for i in range(X_test.shape[1]):
            x = X_test[:,i].reshape((X_test.shape[0], 1))
            distances = self.euclidean_distance(x, self.X)
            k_nearest = np.argpartition(distances, k)
            predicts[:,i] = self.voting(k_nearest, distances, self.y)

        if y_test is not None:  # used for test purpose
            acc = self.calculate_accurracy(predicts, y_test)
            return predicts, acc
        else:
            return predicts