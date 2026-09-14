import numpy as np
from sklearn.preprocessing import StandardScaler

class RegressionModel:
    def __init__(self, data, scale=False, lambda_=0.01,alpha=0.01, num_iters=1000):
        self.scale = scale
        self.cost_history = None
        self.data = data
        self.X, self.y = self.data_preprocessing()
        self.theta = np.zeros(self.X.shape[1])
        self.scaler = None
        self.lambda_ = lambda_
        self.alpha = alpha
        self.num_iters = num_iters

    def data_preprocessing(self):
        m, n = self.data.shape
        X = self.data.iloc[:, :-1].values
        X = np.c_[np.ones((m, 1)), X]
        if self.scale:
            self.scaler = StandardScaler()
            X[:, 1:] = self.scaler.fit_transform(X[:, 1:])
        y = self.data.iloc[:, -1].values
        return X, y

    def naive_cost_function(self):
        """
        This method calculates the cost of the current values of theta, that is, the extent to which the
        prediction (given the specific value of theta) corresponds to the actual value (that
        is given in y).
        :return: cost
        """
        cumulative_cost = 0
        rows, columns = self.data.shape
        for observation_number in range(0, len(self.X)):
            predicted = sum(value * self.theta[index] for index, value in enumerate(self.X[observation_number]))
            cumulative_cost += (predicted - self.y[observation_number]) ** 2
        return cumulative_cost * (1 / (2 * rows))

    def reverse_theta(self):
        """
        Transforms the theta parameters obtained from a regression model on scaled data
        back to the original (non-scaled) feature space.

        Parameters:
        theta (numpy array): Array of theta parameters obtained from the regression model.
        scaler (StandardScaler object): The scaler object used to scale the features.

        Returns:
        numpy array: transformed theta parameters in the context of the original, non-scaled features.

        The function performs the following steps:
        1. Initializes a new array `theta_original` with the same shape as `theta`.
        2. Corrects the intercept (theta[0]) to account for the means of the original features.
        3. Adjusts the coefficients (theta[1:]) by reversing the effect of scaling using the standard deviations.
        """
        # initialize
        theta_original = np.zeros_like(self.theta)
        # tranform back intercept
        theta_original[0] = self.theta[0] - np.sum((self.theta[1:] * self.scaler.mean_) / self.scaler.scale_)
        # transform back coefficients
        theta_original[1:] = self.theta[1:] / self.scaler.scale_

        return theta_original

    def compute_cost(self):
        """
        This method calculates the cost of the current values of theta, that is, the extent to which the
        prediction (given the specific value of theta) corresponds to the actual value (that
        is given in y).

        Every data point in X is multiplied by theta (which dimensions have X and thus theta transposed)
        and the result of this is compared with the actual value (so with y). The difference between
        these two values are squared and the total of all these squares is divided by it
        number of data points to get the average. You must return this average (the variable
        J: a number, in short).
        """

        number_of_observations = len(self.y)
        predicted = self.X @ self.theta.T
        errors = predicted - self.y
        regularisation = (self.lambda_/ number_of_observations)  * np.sum(self.theta[1:] ** 2)
        cost = (1 / (2 * number_of_observations)) * np.sum(errors ** 2) + regularisation
        return cost

    def gradient_descent(self):
        """
        In this problem, every parameter of theta num_iter is updated times the optimal values
        for these parameters. Per iteration you have to update all parameters of theta.

        Each parameter of theta is reduced by the sum of the error of all data points
        multiplied by the data point itself (see the formula above).
        This sum itself is multiplied by the 'learning rate' alpha.

        """
        # initialize list of costs
        cost_history = np.zeros(self.num_iters)
        number_of_observations = len(self.y)
        for index in range(self.num_iters):
            predicted = self.X @ self.theta.T
            errors = predicted - self.y
            gradients = (self.X.T @ errors) / number_of_observations
            # regularisation
            gradients[1:] += (self.lambda_ / number_of_observations) * self.theta[1:]
            self.theta = self.theta - self.alpha * gradients
            cost_history[index] = self.compute_cost()
        return cost_history

    def fit(self):
        self.cost_history = self.gradient_descent()
        return self

    def predict(self, X_new):
        if self.scale:
            X_new = self.scaler.transform(X_new)
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.theta
