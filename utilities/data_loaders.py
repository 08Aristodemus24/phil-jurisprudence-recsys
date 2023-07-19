import numpy as np
import pandas as pd

def load_precalc_params_small():
    file = open('./data/small_movies_X.csv', 'rb')
    X = np.loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = np.loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = np.loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./data/small_movies_Y.csv', 'rb')
    Y = np.loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = np.loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('./data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return (mlist, df)