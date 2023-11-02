import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


class LinearDecisionBoundaryPlotter:
    def __init__(self, X, y, frame_rate):
        self.X = X
        self.y = y
        self.frame_rate = frame_rate
        self.fig = None
        self.ax = None
        self.line = None
        self.x_values = None

    def plot_scatter_with_decision_boundary(self, normal_vector, bias):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')

        for label in np.unique(self.y):
            self.ax.scatter(self.X[self.y.ravel() == label, 0], self.X[self.y.ravel() == label, 1], label=f'Label {label}')

        self.x_values = np.linspace(self.X[:, 0].min() -1, self.X[:, 0].max() +1, 100)
        y_values = (-normal_vector[0]*self.x_values - bias) / normal_vector[1]
        self.line, = self.ax.plot(self.x_values, y_values, '-k', label='Decision Boundary')

        return self.fig, self.ax, self.line

    def update_plot(self, normal_vector, bias):
        y_values = (-normal_vector[0]*self.x_values - bias) / normal_vector[1]
        
        self.line.set_ydata(y_values)
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        display(self.fig)
        clear_output(wait=True)
        
        time.sleep(1/self.frame_rate)
        return self.fig, self.ax, self.line