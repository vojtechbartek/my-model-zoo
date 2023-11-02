import matplotlib.pyplot as plt
import csv
import os
from . import config

def create_dataset_interactive(file_name: str, data_type: str='classification'):
    class DatasetCreator:
        def __init__(self, ax):
            self.ax = ax
            self.points = []
            self.classes = []
            self.current_class = 1
            self.file_path = os.path.join(config.DATA_FILE_PATH, data_type, file_name)

        def on_key(self, event):
            if event.key == '1':
                self.current_class = 1
            elif event.key == '2':
                self.current_class = -1
            else:
                plt.close()
                self.save_data()

        def on_click(self, event):
            if event.inaxes is not None:
                self.points.append((event.xdata, event.ydata))
                self.classes.append(self.current_class)
                color = 'red' if self.current_class == 1 else 'blue'
                self.ax.plot(event.xdata, event.ydata, 'x', color=color)
                plt.draw()

        def save_data(self):
            with open(self.file_path + '.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'class'])
                for point, class_ in zip(self.points, self.classes):
                    writer.writerow([point[0], point[1], class_])

        def on_close(self, event):
            self.save_data()

    fig, ax = plt.subplots()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])

    creator = DatasetCreator(ax)

    fig.canvas.mpl_connect('key_press_event', creator.on_key)
    fig.canvas.mpl_connect('button_press_event', creator.on_click)
    fig.canvas.mpl_connect('close_event', creator.on_close)

    plt.title('Click to add points, press 1 or 2 to change class, press any other key to finish')
    plt.show()
