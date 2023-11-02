"""Implementation of the perceptron algorithm"""

from .model import Model
import numpy as np
from typing import Type
from src.utils.visualization import LinearDecisionBoundaryPlotter
import matplotlib.pyplot as plt
import time

class Perceptron(Model):
    
    def __init__(self):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.last_iteration = None
        self.plotter = LinearDecisionBoundaryPlotter


    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int, visualize: bool=False, frame_rate:int=2) -> Type['Perceptron']:
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
        K = X.shape[1]
        self.X = X
        self.y = y
        self.w = np.ones((K,))
        self.b = 0

        if visualize:
            # initialize visualization plotter
            self.plotter = self.plotter(X,y,frame_rate)
            self.plotter.plot_scatter_with_decision_boundary(self.w, self.b)
            #plt.ion()
            #plt.show()
        
        for i in range(max_iterations):
            wrong_predictions = False
            for x_,y_ in zip(X,y):
                prediction = x_ @ self.w + self.b
                if prediction * y_ <= 0:
                    # update parameters
                    wrong_predictions = True
                    self.w += y_ * x_
                    self.b += y_
                    if visualize:
                        self.plotter.update_plot(self.w, self.b)
                
            if wrong_predictions is False:
                # no mistake, end algorithm
                #if visualize:
                   # plt.ioff()
                self.last_iteration = i
                return self
        
        # reached max_iterations but training is not done
        #plt.ioff()
        return None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X (np.ndarray): numpy array of shape (N_, K). Points to classify.
            
        Returns:
            np.ndarray: numpy array of shape (N_,). Predictions.
        """
        return X @ self.w + self.b
        
