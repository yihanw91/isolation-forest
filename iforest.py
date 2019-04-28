import numpy as np
import pandas as pd


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.height_limit = np.ceil(np.log2(self.sample_size))
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for _ in range(self.n_trees):
            X_sample = X[np.random.randint(
                X.shape[0], size=self.sample_size), :]
            t = IsolationTree(self.height_limit)
            t.fit(X_sample, improved)
            self.trees.append(t)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        avg_len = []
        for instance in X:
            paths = []
            for t in self.trees:
                node = t.root
                while node.left and node.right is not None:
                    if instance[node.split_attr] < node.split_val:
                        node = node.left
                    else:
                        node = node.right
                paths.append(node.path_length)
            avg_len.append(sum(paths) / self.n_trees)
        avg_len = np.array(avg_len).reshape(len(avg_len), 1)
        return avg_len

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_len = self.path_length(X)
        H = np.log(self.sample_size - 1) + 0.5772156649

        scores = []
        for length in avg_len:
            if length > 2:
                C = 2 * H - 2 * (self.sample_size - 1) / self.sample_size
            elif length == 2:
                C = 1
            else:
                C = 0.0000000001
            score = 2 ** -(length / C)
            scores.append(score)
        return np.array(scores)

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        pred = np.where(scores >= threshold, 1, 0)
        return pred

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        pred = self.predict_from_anomaly_scores(scores, threshold)
        return pred


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 1

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if improved:
            self.root = self.fit_(X, 0)
        else:
            self.root = self.fit_improved_(X, 0)
        return self.root

    def fit_(self, X: np.ndarray, height):
        if height >= self.height_limit or len(X) <= 1:
            size = len(X)
            if size > 2:
                height += 2 * (np.log(size - 1) + 0.5772156649) - \
                    2 * (size - 1) / size
            elif size == 2:
                height += 1
            return Node(None, None, None, None, height)

        split_attr = np.random.randint(0, X.shape[1])
        split_val = np.random.uniform(
            min(X[:, split_attr]), max(X[:, split_attr]))
        left_index = X[:, split_attr] < split_val
        X_left = X[left_index]
        X_right = X[np.invert(left_index)]

        node = Node(split_attr, split_val, self.fit_(
            X_left, height+1), self.fit_(X_right, height+1))
        self.n_nodes += 2

        return node

    def fit_improved_(self, X: np.ndarray, height):
        if height >= self.height_limit or len(X) <= 1:
            size = len(X)
            if size > 2:
                height += 2 * (np.log(size - 1) + 0.5772156649) - \
                    2 * (size - 1) / size
            elif size == 2:
                height += 1
            return Node(None, None, None, None, height)

        best_attr = np.random.randint(0, X.shape[1])
        best_val = np.random.uniform(
            min(X[:, best_attr]), max(X[:, best_attr]))
        if height == 0:
            best_loss = len(X) / 2
            for i in range(2):
                split_attr = np.random.randint(0, X.shape[1])
                for j in range(2):
                    split_val = np.random.uniform(
                        min(X[:, split_attr]), max(X[:, split_attr]))
                    left_index = X[:, split_attr] < split_val
                    X_left = X[left_index]
                    X_right = X[np.invert(left_index)]
                    loss = min(len(X_left), len(X_right))
                    if loss < best_loss:
                        best_loss = loss
                        best_attr = split_attr
                        best_val = split_val
                        break

        left_index = X[:, best_attr] < best_val
        X_left = X[left_index]
        X_right = X[np.invert(left_index)]
        node = Node(best_attr, best_val, self.fit_improved_(
            X_left, height+1), self.fit_improved_(X_right, height+1))
        self.n_nodes += 2

        return node


class Node:
    def __init__(self, split_attr, split_val, left, right, path_length=None):
        self.split_attr = split_attr
        self.split_val = split_val
        self.left = left
        self.right = right
        self.path_length = path_length


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    y_scores = np.concatenate((np.array(y).reshape(len(y), 1), scores), axis=1)

    for s in np.linspace(1.0, 0.0, num=101):
        TP = len(y_scores[(y_scores[:, 1] >= s) & (y_scores[:, 0] == 1)])
        FN = len(y_scores[(y_scores[:, 1] < s) & (y_scores[:, 0] == 1)])
        FP = len(y_scores[(y_scores[:, 1] >= s) & (y_scores[:, 0] == 0)])
        TN = len(y_scores[(y_scores[:, 1] < s) & (y_scores[:, 0] == 0)])
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return s, FPR
