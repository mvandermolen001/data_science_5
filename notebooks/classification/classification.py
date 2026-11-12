import  numpy as np
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array


class NaiveBayes(ClassifierMixin, BaseEstimator):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """Calculate mean and stdev matrices size (n_labels, n_features)"""
        X, y = check_X_y(X, y)

        self.labels_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.priors_ = np.unique_counts(y).counts / len(y)
        n_labels, n_features = len(self.labels_), X.shape[1]

        self.mu_ = np.zeros((n_labels, n_features))
        self.sd_ = np.zeros((n_labels, n_features))
        for i, label in enumerate(self.labels_):
            self.mu_[i] = X[y == label].mean(axis=0)
            # Prevents issues later on, in case a feature with zero variance comes in
            # It is best to just make sure that no such features come in.
            self.sd_[i] = np.maximum(X[y == label].std(axis=0), self.var_smoothing)
        return self

    def predict(self, data):
        """
        Classify each test sample

        Args:
            - data: Test data samples, a matrix of dimensions (m, n)

        Returns:
            - y_pred: List of predicted class labels for each test sample
        """
        posteriors = []
        y_pred = []
        for row in data:
            for i, label in enumerate(self.labels_):
                diff = np.square(row - self.mu_[i])
                norm_factors = 1 / (np.sqrt(2 * np.pi) * self.sd_[i])
                likelihoods = norm_factors * np.exp(-diff / (2 * self.sd_[i] ** 2))
                posteriors.append(np.prod(likelihoods) * self.priors_[i])
            y_pred.append(self.labels_[np.argmax(posteriors)])
            posteriors = []
        return y_pred

class ID3(ClassifierMixin, BaseEstimator):
    def __init__(self, max_depth=None, gain_threshold=1e-3):
        self.max_depth = max_depth
        self.gain_threshold = gain_threshold

    @staticmethod
    def entropy(y):
        """
        Calculate the entropy of a distribution for the classes in the target variable (y).
        :return: the entropy score
        """
        # probs are the frequency of each class relative to the total number of labels (classes)
        # Entropy of the target (Vital Status) using sample.txt (no splitting) (using log2 as a base): 0.97
        probs = np.unique_counts(y).counts / len(y)
        entropy_score = -sum(probs * np.log2(probs))
        return entropy_score

    def information_gain(self, X, feature, target):
        """
        Calculate the information gain of the target variable (y).
        :param feature: the feature to calculate the information gain
        :param target: the target variable
        :return: information gain when taking the feature information
        """
        IG = self.entropy(target)
        for val in np.unique(X[:, feature]):
            mask = X[:, feature] == val
            subdata = X[mask, :]
            IG -= len(subdata) / len(X) * self.entropy(target[mask, :])
        return IG

    def id3(self, X, y, depth=0):
        """
        Creating the tree, using id3 algorithm.
        ::param X: a numpy matrix of dimensions (m, n)
        :param y: a numpy matrix with target variable
        :param target: the target variable
        :param depth: Current depth
        :param max_depth: how many iterations / leafs to construct
        :return: a fitted tree
        """
        gain, best_feature = max(((self.information_gain(X, feature, y), feature)
                                  for feature in range(X.shape[1])))
        result = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            mask = X[:, best_feature] == value
            subdata = X[mask, :]
            sub_y = y[mask, :]
            if (self.max_depth is not None and depth >= self.max_depth) or gain < self.gain_threshold:
                # BASE CASE
                subresult = sub_y.mean(axis=0)
            else:
                # RECURSIVE CASE
                subresult = self.id3(subdata, sub_y, depth + 1)
            # COMBINATION
            result[best_feature][value] = subresult
        return result

    def fit(self, X, y):
        """Fitting a tree using the id3 algorithm.
        :param X: a numpy matrix of dimensions (m, n)
        :param y: a numpy matrix with target variable
        :param features: a list of features use from the matrix"""
        self.classes_ = unique_labels(y)
        onehot = y[:, np.newaxis] == self.classes_[np.newaxis]
        self.tree_ = self.id3(X, onehot, 0)
        return self

    def predict_proba(self, X):
        result = []
        for sample in X:
            tree = self.tree_
            while True:
                feature = list(tree.keys())[0]
                value = sample[feature]
                try:
                    tree = tree[feature][value]
                except KeyError:
                    result.append([np.nan] * len(self.classes_))
                    break
                else:
                    if not isinstance(tree, dict):
                        result.append(tree)
                        break
        return np.array(result)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        probas = self.predict_proba(X)
        idxs = np.argmax(probas, axis=1)
        return self.classes_[idxs]