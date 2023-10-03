import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import seaborn as sb
import networkx as nx

import pandas as pd
import tensorflow as tf



def view_tensor_values(tensor):
    """
    converts the given keras tensor to numpy array
    """
    return tf.convert_to_tensor(tensor)


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
    # print(unique_counts)

    unique_ids = df[col].unique()
    print(f'total unique values: {len(unique_ids)}')
    # print(unique_ids)

def train_cross_results_v2(results: dict, epochs: list, curr_metrics_indeces: tuple, img_title: str='figure', image_only=False):
    """
    plots the number of epochs against the cost given cost values across these epochs
    """

    # use matplotlib backend
    mpl.use('Agg')

    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [
        ('p:', '#f54949'), 
        ('h-', '#f59a45'), 
        ('o--', '#afb809'), 
        ('x:','#51ad00'), 
        ('+:', '#03a65d'), 
        ('8-', '#035aa6'), 
        ('.--', '#03078a'), 
        ('>:', '#6902e6'),
        ('p-', '#c005e6'),
        ('h--', '#fa69a3'),
        ('o:', '#240511'),
        ('x-', '#052224'),
        ('+--', '#402708'),
        ('8:', '#000000')]

    for index, (key, value) in enumerate(results.items()):
        if key == "loss" or key == "val_loss":
            # e.g. loss, val_loss has indeces 0 and 1
            # binary_cross_entropy, val_binary_cross_entropy 
            # has indeces 2 and 3
            axis.plot(np.arange(len(epochs)), value, styles[curr_metrics_indeces[index]][0], color=styles[curr_metrics_indeces[index]][1], alpha=0.5, label=key)
        else:
            metric_perc = [round(val * 100, 2) for val in value]
            axis.plot(np.arange(len(epochs)), metric_perc, styles[curr_metrics_indeces[index]][0], color=styles[curr_metrics_indeces[index]][1], alpha=0.5, label=key)

    # annotate end of lines
    for index, (key, value) in enumerate(results.items()):        
        if key == "loss" or key == "val_loss":
            last_loss_rounded = round(value[-1], 2)
            axis.annotate(last_loss_rounded, xy=(epochs[-1], value[-1]), color=styles[curr_metrics_indeces[index]][1])
        else: 
            last_metric_perc = round(value[-1] * 100, 2)
            axis.annotate(last_metric_perc, xy=(epochs[-1], value[-1] * 100), color=styles[curr_metrics_indeces[index]][1])

    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.set_title(img_title)
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')

    if image_only is False:
        plt.show()

    # delete figure
    del figure

def view_value_frequency(value_counts:pd.Series, colormap:str, title: str, kind: str='barh', limit: int=6, order='ASC'):
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
    # print(data)

    cmap = cm.get_cmap(colormap)
    fig = plt.figure(figsize=(15, 10))
    axis = fig.subplots()

    
    
    if kind == 'barh':        
        axis.barh([str(index) for index in data.index], data.values, color=cmap(np.linspace(0, 1, len(data))))
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
    """
    args:
        kg - is the knowledge graph represented as a dataframe with columns head, relation, tail
        which are the triples that make up the knowledge graph
    """
    # see first 500 rows
    G = nx.from_pandas_edgelist(kg[:limit].loc[kg['relation'] == edge], source='head', target='tail', edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G, k=0.5)

    plt.figure(figsize=(12, 12))

    nx.draw(G, with_labels=True, node_color=node_color, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()