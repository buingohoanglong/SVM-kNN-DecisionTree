from functools import *
import math
import numpy as np

class Node:
    def __init__(self, ids, depth):
        self.ids = ids # index of datapoint in dataset
        self.entropy = None

        # leaf node attribute
        self.label = None

        # internal node attribute
        self.split_attribute = None # index of attribute to split at this node
        self.split_value = None # value to be split at self.slit_attribute
        self.children = []  # list of child nodes (left and right)

        self.depth = depth

    def set_state(self, split_attribute, split_value, children):
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.children = children




class DecisionTree:
    def __init__(self, max_depth):
        self.X = None # list of data point, each data point is a list of attributes (write in row)
        self.y = None  # list of labels

        self.root = None

        self.max_depth = max_depth


    def entropy(self, node):
        labels, counts = np.unique(self.y[:,node.ids], return_counts=True)
        probs = counts / node.ids.shape[0]
        node.entropy = 0 if np.any(probs == 0) else -np.sum(probs*np.log(probs))
        return node.entropy


    def set_label(self, node):
        labels, counts = np.unique(self.y[:,node.ids], return_counts=True)
        node.label = labels[counts.argmax()]


    def split(self, node, attribute):
        split_value = None
        best_information_gain = 0
        children = []

        X_concat = np.concatenate((self.X[node.ids],self.y[:,node.ids].T,node.ids.reshape((1,-1)).T), axis=1)
        X_concat_sorted = X_concat[np.argsort(X_concat[:, attribute])]

        split_value_set = np.unique(X_concat_sorted[:,attribute])
        for value in split_value_set:
            i = X_concat_sorted[:,attribute].searchsorted(value)
            left_ids, right_ids = X_concat_sorted[:i,-1].astype(int), X_concat_sorted[i:,-1].astype(int)
            left_child = Node(ids=left_ids, depth=node.depth+1)
            right_child = Node(ids=right_ids, depth=node.depth+1)

            if left_child.ids.shape[0] == node.ids.shape[0] or right_child.ids.shape[0] == node.ids.shape[0]:
                information_gain = 0
            else:
                information_gain = node.entropy - (left_child.ids.shape[0]*self.entropy(left_child) + right_child.ids.shape[0]*self.entropy(right_child)) / node.ids.shape[0]

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                children = [left_child, right_child]
                split_value = value

        return best_information_gain, attribute, split_value, children



    def split_node(self, node):
        best_information_gain = 0
        split_attribute = None
        split_value = None
        children = []

        for attribute in range(self.X.shape[1]):
            information_gain, current_split_attribute, current_split_value, current_children = self.split(node, attribute)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                split_attribute = current_split_attribute
                split_value = current_split_value
                children = current_children

        node.set_state(split_attribute, split_value, children)


    def build_tree(self):
        # build tree in bfs manner
        queue = [self.root]
        while len(queue) != 0:
            node = queue.pop(0)
            if self.entropy(node) == 0 or node.depth >= self.max_depth: # leaf node
                self.set_label(node)
            else: # internal node
                self.split_node(node)
                if node.children == []:
                    self.set_label(node)
                else:
                    for child in node.children:
                        queue.append(child)


    def fit(self, X, y):
        self.X = X.T # (num data, num attr)
        self.y = y  # (1, num data)

        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

        ids = np.arange(self.y.shape[1], dtype=int)
        print(f"ids shape: {ids.shape}")
        self.root = Node(ids=ids, depth=0)
        
        self.build_tree()


    def __predict(self, x, node):
        if node.label is not None:
            return node.label

        if x[node.split_attribute] >= node.split_value:
            return self.__predict(x, node.children[1])
        else:
            return self.__predict(x, node.children[0])


    def predict(self, x):
        return self.__predict(x, self.root)

    def test(self, X, y):
        X_test = X.T
        predicts = np.zeros(y.shape)

        for index in range(X_test.shape[0]):
            x = X_test[index]
            predicts[0, index] = self.predict(x)

        # calculate accurracy
        f = np.frompyfunc(lambda x: x if x == 1 else 0,1,1)
        num_correct_predicts = np.sum(f(predicts*y))
        num_predicts = y.shape[1]
        acc = num_correct_predicts / num_predicts

        return predicts, acc