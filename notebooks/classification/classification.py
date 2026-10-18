import  numpy as np

class NaiveBayes:
    def __init__(self, X, y):
        self.sd, self.mu = None, None
        self.X = X
        self.y = y
        self.labels = np.unique(self.y)
        self.priors = np.unique_counts(self.y).counts / len(self.y)

    def fit(self):
        """Calculate mean and stdev matrices size (n_labels, n_features)"""
        n_labels, n_features = len(self.labels), self.X.shape[1]
        self.mu = np.zeros((n_labels, n_features))
        self.sd = np.zeros((n_labels, n_features))
        for i, label in enumerate(self.labels):
            self.mu[i] = self.X[self.y == label].mean(axis=0)
            # Prevents issues later on, in case a feature with zero variance comes in
            # It is best to just make sure that no such features come in.
            self.sd[i] = np.maximum(self.X[self.y == label].std(axis=0), 1e-9)
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
            for i, label in enumerate(self.labels):
                diff = np.square(row - self.mu[i])
                norm_factors = 1 / (np.sqrt(2 * np.pi) * self.sd[i])
                likelihoods = norm_factors * np.exp(-diff / (2 * self.sd[i] ** 2))
                posteriors.append(np.prod(likelihoods) * self.priors[i])
            y_pred.append(self.labels[np.argmax(posteriors)])
            posteriors = []
        return y_pred

class ID3:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

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

    @staticmethod
    def majority_label(y):
        """
        Get the majority label of the target variable (y).
        :param y: a list-like with labels
        :return: the majority label
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def information_gain(self, feature, target):
        """
        Calculate the information gain of the target variable (y).
        :param feature: the feature to calculate the information gain
        :param target: the target variable
        :return: information gain when taking the feature information
        """
        before = self.entropy(self.X[target])
        after = 0
        for value, subset in self.X.groupby(feature):
            weight = len(subset) / len(self.X)
            after += weight * self.entropy(subset[target])
        return before - after

    def id3(self, data, features, target, depth=0, max_depth=None):
        """
        Creating the tree, using id3 algorithm.
        :param data: a pandas dataframe of dimensions (m, n)
        :param features: the features use from the matrix
        :param target: the target variable
        :param depth:
        :param max_depth: how many iterations / leafs to construct
        :return: a fitted tree
        """
        y = data[target]
        if len(y.unique()) == 1:
            return y.iloc[0]
        # ID3 algorithm to create a decision tree with max depth
        gains = [self.information_gain(f, y) for f in features]
        best_idx = int(np.argmax(gains))
        best_feature = features[best_idx]
        if len(features) == 0 or (max_depth is not None and depth >= max_depth):
            return self.majority_label(y)
        if gains[best_idx] <= 0:
            return self.majority_label(y)
        tree = {best_feature: {}}
        remaining = [f for f in features if f != best_feature]
        for value, subset in data.groupby(best_feature):
            subtree = self.id3(subset, remaining, target,
                          depth=depth + 1, max_depth=max_depth)
            tree[best_feature][value] = subtree
        return tree

    def fit(self, X, y, features):
        """Fitting a tree using the id3 algorithm.
        :param X: a pandas dataframe of dimensions (m, n)
        :param y: a pandas dataframe with the target variable
        :param features: a list of features use from the matrix"""
        self.tree = self.id3(X, features, y, depth=0, max_depth=self.max_depth)
        return self

    def predict_one(self, sample):
        """
        Predicts a sample using the fitted tree
        :param sample: a sample with the needed features
        :return: a predicted label
        """
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            value = sample[feature]
            node = node[feature][value]
        return node

    def predict_all(self, samples):
        """
        Predicts all the samples using the fitted tree
        :param samples: some dataframe of dimensions (m, n)
        :return: a list of predicted labels
        """
        return [self.predict_one(row) for _, row in samples.iterrows()]