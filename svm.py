import numpy as np

class SVM:
    def __init__(self):
        self.weights = None

    def calculate_accurracy(self, predicts, y):
        f = np.frompyfunc(lambda x: x if x == 1 else 0,1,1)
        num_correct_predicts = np.sum(f(predicts*y))
        num_predicts = y.shape[1]
        acc = num_correct_predicts / num_predicts
        return acc

    def has_positive_hinge_loss(self, w_bar, x_bar, y):
        f = np.frompyfunc(lambda x: 1 if x > 0 else 0, 1, 1)
        return f(1 - y*w_bar.T.dot(x_bar))

    def fit(self, X, y, learning_rate=0.001, ld=1, loops=1000):
        X_bar = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0) # extend
        self.weights = np.ones((X_bar.shape[0], 1)) # extend

        for i in range(loops):
            if i%100 == 0:
                predicts, acc = self.predict(X,y)
                print(f"Acc: {acc}")
        
            temp = self.has_positive_hinge_loss(self.weights, X_bar, y)
            regularization_gradient = ld * np.concatenate((self.weights[:-1], np.array([[0]])), axis=0)
            hinge_gradient = np.sum(-temp*y*X_bar, axis=1).reshape(X_bar.shape[0], 1)
            self.weights = self.weights - learning_rate * ( hinge_gradient + regularization_gradient )


    def predict(self, X, y=None):
        X_bar = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0) # extend
        predicts = np.sign(self.weights.T.dot(X_bar))

        if y is not None: # used for test purpose
            # calculate accurracy
            acc = self.calculate_accurracy(predicts, y)
            return predicts, acc
        else:
            return predicts

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            np.save(f, self.weights)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self.weights = np.load(f, allow_pickle=True)
  