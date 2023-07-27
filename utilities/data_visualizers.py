import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import seaborn as sb
import networkx as nx

import pandas as pd


def view_vars(Y, R, X, THETA, BETA):
    """
    args:
        Y - user-item rating matrix
        R - user-item interaction matrix
        X - item embedding matrix or the learned parameters/embeddings from user-item rating matrix Y
        THETA - user embedding matrix or the learned parametes from user-item rating matrix Y
        BETA - user bias embedding vector learned from user-item rating matrix Y
    """
    print(f'Y: {Y} shape: {Y.shape}\n')
    print(f'R: {R} shape: {R.shape}\n')
    print(f'X: {X} shape: {X.shape}\n')
    print(f'THETA: {THETA} shape: {THETA.shape}\n')
    print(f'BETA: {BETA} shape: {BETA.shape}\n')


def describe_col(df, col):
    """
    args:
        df - pandas data frame
        col - column of data frame to observe unique values and frequency of each unique value
    """

    unique_counts = df[col].value_counts()
    print(f'count/no. of occurences of each unique {col} out of {df.shape[0]}: \n')
    print(unique_counts)

    unique_ids = df[col].unique()
    print(unique_ids)
    print(f'total unique values: {len(unique_ids)}')

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

def view_value_frequency(value_counts:pd.Series, colormap:str, title: str, kind: str='barh', limit: int=6):
    """
    plots either a horizontal bar graph to display frequency of words top 'limit' 
    words e.g. top 20 or a pie chart to display the percentages of the top 'limit' 
    words e.g. top 20, specified by the argument kind which can be either
    strings barh or pie

    args: 
        value_counts - 
        colormap - 
        title - 
        kind - 
        limit - 
    """

    # get either last few words or first feww words
    data = value_counts.sort_values(ascending=True)[:limit]
    # data = value_counts.sort_index(ascending=False)[:limit]
    print(data)

    cmap = cm.get_cmap(colormap)
    fig = plt.figure(figsize=(15, 10))
    axis = fig.subplots()

    
    
    if kind == 'barh':        
        axis.barh(data.index, data.values, color=cmap(np.linspace(0, 1, len(data))))
        # axis = value_counts[:limit].sort_values(ascending=True).plot(kind='barh', colormap='viridis')
        
        axis.set_xlabel('frequency')
        axis.set_ylabel('words')
        axis.set_title(title)
        plt.savefig(f'./figures & images/{title}.png')

        plt.show()
    elif kind == 'pie':
        axis.pie(data, labels=data.index, autopct='%.2f%%', colors=cmap(np.linspace(0, 1, len(data))))
        axis.axis('equal')
        axis.set_title(title)
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()


def visualize_graph(kg, limit: str=500, edge: str='film.film.genre', node_color: str='skyblue'):
    # see first 500 rows
    G = nx.from_pandas_edgelist(kg[:limit].loc[kg['relation'] == edge], source='head', target='tail', edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G, k=0.5)

    plt.figure(figsize=(12, 12))

    nx.draw(G, with_labels=True, node_color=node_color, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()