from typing import Type
import numpy as np
from .model import Model
from src.utils.visualization import LinearDecisionBoundaryPlotter


class Perceptron(Model):
    def __init__(self, learning_rate: float = 1):
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.steps = None
        self.plotter = None
        self.X = None
        self.y = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int,
        visualize: bool = False,
        frame_rate: int = 2,
    ) -> Type["Perceptron"]:
        """
        Train model given training dataset.
        Perceptron training algorithm is as follows:
        Let X be a vector of K-dimensional points in space R^K and y be a vector
        of associated labels {-1, +1}.
        We then combine these two vectors into vector D, by appending the y_i
        to the end of X_i (D_i = [X_i1, X_i2,..., X_iK, y_i]), this simplifies the
        following training process:
        We create a zero vector of dimension (K+1, 1) called Theta.
        Now we iterate for up to max_iteration number of steps and at each step
        we go through all D_i points and we try if the dot product between D_i and
        Theta is greater than zero, if so, the classification is correct and we
        continue to the next point D_i+1. If not, classification is incorrect and we
        perform an update of our parameters Theta by adding the incorectly
        classified vector D_i to Theta. We repeat this until we make no errors or
        until we reach max_iterations

        Args:
            X (np.ndarray): numpy array of training points of shape (N, K).
            y (np.ndarray): numpy array of labels of shape (N, 1).
            visualize (bool): True if we want to plot every train step.
            frame_rate (int): Frame rate for visualizaiton.
        Returns:
            class object fitted (trained) on the given training set.
        """
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.steps = 0
        self.X = X
        self.y = y

        if visualize:
            # initialize visualization plotter
            self.plotter = LinearDecisionBoundaryPlotter(X, y, frame_rate)
            self.plotter.plot_scatter_with_decision_boundary(self.w, self.b)

        for _ in range(max_iterations):
            wrong_predictions = False
            for x_, y_ in zip(X, y):
                if self.predict(x_) * y_ <= 0:
                    # update parameters
                    self.w += self.learning_rate * y_ * x_
                    self.b += self.learning_rate * y_

                    self.steps += 1
                    wrong_predictions = True
                    if visualize:
                        self.plotter.update_plot(self.w, self.b)

            if not wrong_predictions:
                # no mistake, end algorithm
                return self

        # reached max_iterations but training is not done
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X (np.ndarray): numpy array of shape (N_, K). Points to classify.

        Returns:
            np.ndarray: numpy array of shape (N_,). Predictions.
        """
        return np.sign(X @ self.w + self.b)
