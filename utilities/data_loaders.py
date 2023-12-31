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
    
def load_raw_movie_1m_ratings(dir_path):
    """
    returns df with columns of users, the items they interacted with, 
    their rating of that item, and the timestamp in which they did so
    """

    df = pd.read_csv(f'{dir_path}', sep='::', header=None)
    df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating', 3: 'timestamp'}, inplace=True)

    return df

def load_raw_movie_20k_kg(dir_path):
    """
    returns the knowledge graph as a dataframe of 3 columns representing
    each triple in the knowledge graph
    """

    kg = pd.read_csv(f'{dir_path}/kg.txt', sep='\t', header=None)
    kg.rename(columns={0: 'head', 1: 'relation', 2: 'tail'}, inplace=True)

    return kg
    
def load_raw_juris_300k_ratings(dir_path: str):
    """
    returns a the .csv file of the synthesized jurisprudence document data
    set. Contains the columns of users, the items they "interacted with",
    and the synthetic rating created for that user-item interaction
    """

    df = pd.read_csv(f'{dir_path}', index_col=0)
    
    return df

def load_raw_juris_600k_ratings(dir_path: str):
    """
    returns a the .csv file of the synthesized jurisprudence document data
    set. Contains the columns of users, the items they "interacted with",
    and the synthetic rating created for that user-item interaction
    """

    df = pd.read_csv(f'{dir_path}', index_col=0)
    
    return df


def load_raw_juris_ratings(dir_path: str):
    """
    returns a the .csv file of the synthesized jurisprudence document data
    set. Contains the columns of users, the items they "interacted with",
    and the synthetic rating created for that user-item interaction
    """

    df = pd.read_csv(f'{dir_path}', delim_whitespace=True, header=None)
    df.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)
    
    return df


def load_item_index2entity_id_file(dir_path: str):
    """
    returns the dataframe of the mappings of item_id's in the raw rating file
    to the head entities in the knowledge graph
    """
    
    item_index2entity_id = pd.read_csv(f'{dir_path}/item_index2entity_id.txt', sep='\t', header=None)
    item_index2entity_id.rename(columns={0: 'item_index', 1: 'entity_id'}, inplace=True)

    return item_index2entity_id

def load_final_movie_1m_ratings(dir_path: str):
    """
    returns the dataframe of the final preprocessed movielens 1m rating
    file
    """

    ratings_final = pd.read_csv(f'{dir_path}/ratings_final.txt', sep='\t', header=None)
    ratings_final.rename(columns={0: 'user_id', 1: 'item_id', 2: 'interaction'}, inplace=True)

    return ratings_final
    

def load_data_splits(dataset: str, dir_path: str):
    """
    since all choices of data to use will be preprocessed first
    and then have an output .csv file this function will returns
    the split final preprocessed data
    """

    out_file = dataset.replace('-', '_')
    train_data = pd.read_csv(f'{dir_path}/{out_file}_train.csv', index_col=0)
    cross_data = pd.read_csv(f'{dir_path}/{out_file}_cross.csv', index_col=0)
    test_data = pd.read_csv(f'{dir_path}/{out_file}_test.csv', index_col=0)

    return train_data, cross_data, test_data

