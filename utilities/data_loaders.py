import numpy as np
import pandas as pd

def load_precalc_params_small(dir_path):
    file = open(f'{dir_path}/small_movies_X.csv', 'rb')
    X = np.loadtxt(file, delimiter=",")

    file = open(f'{dir_path}/small_movies_W.csv', 'rb')
    W = np.loadtxt(file, delimiter=",")

    file = open(f'{dir_path}/small_movies_b.csv', 'rb')
    b = np.loadtxt(file, delimiter=",")
    b = b.reshape(1, -1)
    num_movies, num_features = X.shape
    num_users, _ = W.shape
    return (X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small(dir_path):
    file = open(f'{dir_path}/small_movies_Y.csv', 'rb')
    Y = np.loadtxt(file, delimiter=",")

    file = open(f'{dir_path}/small_movies_R.csv', 'rb')
    R = np.loadtxt(file, delimiter=",")
    return (Y, R)

def load_Movie_List_pd(dir_path):
    """ returns df with and index of movies in the order they are in in the Y matrix """

    df = pd.read_csv(f'{dir_path}/small_movie_list.csv', header=None, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return (mlist, df)
    
def load_raw_ratings_large(dir_path):
    file = open(f'{dir_path}/ratings_final.txt', 'rb')
    df = pd.read_csv(file, delimiter='\t', header=None)
    df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)

    return df

    