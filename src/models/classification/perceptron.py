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
        max_iterations: int = 100,
        visualize: bool = False,
        frame_rate: int = 2,
    ) -> Type["Perceptron"]:
        """
        Train model given training dataset.

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
