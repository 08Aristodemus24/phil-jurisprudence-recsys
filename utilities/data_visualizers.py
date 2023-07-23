import numpy as np
import matplotlib.pyplot as plt

def view_vars(Y, R, X, THETA, BETA):
    print(f'Y: {Y} shape: {Y.shape}\n')
    print(f'R: {R} shape: {R.shape}\n')
    print(f'X: {X} shape: {X.shape}\n')
    print(f'THETA: {THETA} shape: {THETA.shape}\n')
    print(f'BETA: {BETA} shape: {BETA.shape}\n')

def train_cross_results_v2(results: dict, epochs: list, img_title: str='figure'):
    """
    plots the number of epochs against the cost given cost values across these epochs
    """
    # # use matplotlib backend
    # mpl.use('Agg')

    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]

    for index, (key, value) in enumerate(results.items()):
        axis.plot(np.arange(len(epochs)), value, styles[index][0] ,color=styles[index][1], alpha=0.5, label=key)

    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.set_title(img_title)
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')
    plt.show()
    

    # delete figure
    del figure