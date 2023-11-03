import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
from src.utils import config

_OFFSET = 5

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


class LinearDecisionBoundaryPlotter:
    def __init__(self, X, y, frame_rate):
        self.X = X
        self.y = y
        self.frame_rate = frame_rate
        self.fig = None
        self.ax = None
        self.line = None
        self.x_values = None
        self.norm_arrow = None

    def plot_scatter_with_decision_boundary(self, normal_vector, bias):
        self.fig, self.ax = plt.subplots(figsize=config.FIG_SIZE)
        # make axis equal
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")

        for label in np.unique(self.y):
            self.ax.scatter(
                self.X[self.y.ravel() == label, 0],
                self.X[self.y.ravel() == label, 1],
                label=f"Label {label}",
            )

        self.x_values = np.linspace(
            self.X[:, 0].min() - _OFFSET, self.X[:, 0].max() + _OFFSET, 100
        )

        # initialize decision boundary line
        (self.line,) = self.ax.plot(
            self.x_values, np.zeros_like(self.x_values), "-k", label="Decision Boundary"
        )

        # Add an arrow representing the normal vector
        self.norm_arrow = self.ax.quiver(
            0,
            0,
            0,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#E03E92",
            label="Normal Vector",
        )

        self.ax.legend()
        display(self.fig)
        clear_output(wait=True)
        time.sleep(1 / self.frame_rate)

        return self.fig, self.ax, self.line

    def update_plot(self, normal_vector, bias):
        if normal_vector[1] == 0:
            if normal_vector[0] == 0:
                return
            self.line.set_xdata([-bias / normal_vector[0]])
            self.line.set_ydata(
                [self.X[:, 1].min() - _OFFSET, self.X[:, 1].max() + _OFFSET]
            )
        else:
            y_values = (-normal_vector[0] * self.x_values - bias) / normal_vector[1]
            self.line.set_xdata(self.x_values)
            self.line.set_ydata(y_values)

        print(normal_vector, bias)

        arrow_x = self.X[:, 0].max()
        arrow_y = (-normal_vector[0] * arrow_x - bias) / normal_vector[1]
        self.norm_arrow.set_UVC(
            2 * normal_vector[0] / np.linalg.norm(normal_vector),
            2 * normal_vector[1] / np.linalg.norm(normal_vector),
        )
        self.norm_arrow.set_offsets((arrow_x, arrow_y))

        display(self.fig)
        clear_output(wait=True)

        time.sleep(1 / self.frame_rate)
        return self.fig, self.ax, self.line
