import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class Regression(BaseEstimator):
    """
    Base class for regression. For regression, the following methods must exist:
    - fit
    - predict

    The Regression class has the following attributes:
    X: The feature matrix
    y: The true labels
    scale(boolean): whether to scale the data or not
    lambda_(float): the lambda parameter for regularization
    alpha(float): the alpha parameter for the learning rate
    num_iters(int): number of iterations to optimize the model with
    """

    def __init__(self,lambda_=0.01,alpha=0.01,num_iters=1000):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.num_iters = num_iters

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, new_X):
        raise NotImplementedError

class LinearRegression(RegressorMixin, Regression):
    def __init__(self, lambda_=0.01, alpha=0.01, num_iters=1000):
        super().__init__(lambda_, alpha, num_iters)

    @staticmethod
    def naive_cost_function(X, y, theta):
        """
        This method calculates the cost of the current values of theta, that is, the extent to which the
        prediction (given the specific value of theta) corresponds to the actual value (that
        is given in y).
        :return: cost
        """
        cumulative_cost = 0
        rows, columns = X.shape
        for observation_number in range(0, len(X)):
            predicted = sum(value * theta[index] for index, value in enumerate(X[observation_number]))
            cumulative_cost += (predicted - y[observation_number]) ** 2
        return cumulative_cost * (1 / (2 * rows))

    def reverse_theta(self, scaler):
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
        theta_original = np.zeros_like(self.theta_)
        # tranform back intercept
        theta_original[0] = self.theta_[0] - np.sum((self.theta_[1:] * scaler.mean_) / scaler.scale_)
        # transform back coefficients
        theta_original[1:] = self.theta_[1:] / scaler.scale_
        return theta_original

    def compute_cost(self, X ,y):
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

        number_of_observations = len(y)
        predicted = X @ self.coef_ + self.intercept_
        errors = predicted - y
        regularisation = self.lambda_ * np.sum(self.coef_ ** 2)
        cost = (1 / (2 * number_of_observations)) * (np.sum(errors ** 2) + regularisation)
        return cost

    def gradient_descent(self, X ,y):
        """
        In this problem, every parameter of theta num_iter is updated times the optimal values
        for these parameters. Per iteration, you have to update all parameters of theta.

        Each parameter of theta is reduced by the sum of the error of all data points
        multiplied by the data point itself (see the formula above).
        This sum itself is multiplied by the 'learning rate' alpha.

        """
        # initialize list of costs
        cost_history = np.zeros(self.num_iters)
        number_of_observations = len(y)
        for index in range(self.num_iters):
            predicted = X @ self.coef_ + self.intercept_
            errors = predicted - y
            gradient = (X.T @ errors) / number_of_observations
            intercept_gradient = np.mean(errors)
            # regularisation
            gradient += (self.lambda_ / number_of_observations) * self.coef_
            self.coef_ -= self.alpha * gradient
            self.intercept_ -= self.alpha * intercept_gradient
            cost_history[index] = self.compute_cost(X, y)
        return cost_history

    def fit(self, X ,y):
        X, y = np.asarray(X), np.asarray(y)
        X, y = check_X_y(X, y, y_numeric=True)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0


        self.cost_history_ = self.gradient_descent(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, new_X):
        check_is_fitted(self, ["coef_", "intercept_"])
        new_X = check_array(new_X)
        return new_X @ self.coef_ + self.intercept_

class LogisticRegression(ClassifierMixin, Regression):
    @staticmethod
    def sigmoid(z):
        """Sigmoid function
        Parameters:
            z: numpy array
        Returns:
            An array with z fitted to the sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """
        Compute cost for logistic regression with regularization.

        Parameters:
        X:  Input feature matrix (m x n)
        y: True labels vector (m,)
        theta: Parameters vector (n,)
        lambda_: Regularization parameter

        Return:
        J:Cost value
        """
        number_of_observations = len(y)
        z = X @ self.coef_ + self.intercept_
        h = self.sigmoid(z)
        regularisation = (self.lambda_ * np.sum(self.coef_ ** 2)) / (2 * number_of_observations)
        cost = (-1 / number_of_observations) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + regularisation
        return cost

    def gradient_descent(self, X, y):
        """
        Perform gradient descent to find optimal theta.

        Parameters:
        X: Input feature matrix (m x n)
        y: True labels vector (m,)
        theta: Initial parameters vector (n,)
        alpha: Learning rate
        num_iters: Number of iterations
        lambda_:  Regularization parameter

        Return:
        theta:  Updated parameters vector
        J_history:  History of cost values
        """
        cost_history = np.zeros(self.num_iters)
        number_of_observations = len(y)
        for index in range(self.num_iters):
            z = X @ self.coef_ + self.intercept_
            h = self.sigmoid(z)
            gradient = (X.T @ (h - y)) / number_of_observations
            intercept_gradient = np.mean(h - y)
            # regularisation
            gradient += (self.lambda_ / number_of_observations) * self.coef_
            # updating
            self.coef_ -= self.alpha * gradient
            self.intercept_ -= self.alpha * intercept_gradient
            cost_history[index] = self.compute_cost(X, y)
        return cost_history

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.cost_history_ = self.gradient_descent(X, y)
        return self

    def predict(self, new_X, threshold=0.5):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters theta.

        Parameters:
        X: Input feature matrix (m x n)
        theta: Parameters vector (n,) (computed by compute cost and gradient descentl)
        threshold: Threshold for prediction

        Return:
        p: Predicted labels vector (m,)
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(new_X)
        probabilities = self.sigmoid( X @ self.coef_ + self.intercept_)
        return np.where(probabilities >= threshold,self.classes_[1],self.classes_[0])

    def predict_proba(self, new_X):
        """Predict probabilities using learned logistic regression parameters theta.
        Parameters:
        new_X: Input feature matrix (m x n)
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(new_X)
        probabilities = self.sigmoid( X @ self.coef_ + self.intercept_)
        return probabilities