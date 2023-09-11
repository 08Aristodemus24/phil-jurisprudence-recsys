import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (get_length__build_value_to_index, 
separate_pos_neg_ratings,
refactor_raw_ratings,
read_item_index_to_entity_id_file, 
convert_rating, 
convert_kg)

from utilities.data_loaders import load_raw_juris_ratings_large, load_raw_movie_ratings_large, load_raw_kg_20k, load_item_index2entity_id_file

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError


if __name__ == "__main__":
    # dataset to choose from
    dataset = {
        'juris-300k': load_raw_juris_ratings_large('./data/juris-300k'),
        'ml-1m': load_raw_movie_ratings_large('./data/ml-1m')
    }

    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, default="juris-300k", help='dataset to use which can be juris-300k for the juris docs rating dataset or ml-1m for the movie lens rating dataset')
    args = parser.parse_args()

    # load user-item rating dataset
    data = dataset[args.d]
    print(data)

    # knowing number of total users and items will be skipped
    # for now just use resulting mapping to preprocess df
    _, user_to_index = get_length__build_value_to_index(data, 'user_id', show_logs=False)
    _, item_to_index = get_length__build_value_to_index(data, 'item_id', show_logs=False)

    # convert old id's to new id's
    data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
    data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

    # separate positive and negative ratings
    pos_user_ratings, neg_user_ratings = separate_pos_neg_ratings(data)

    # finally sample unrated items as our negative class
    refactored_data = refactor_raw_ratings(pos_user_ratings=pos_user_ratings, neg_user_ratings=neg_user_ratings, item_to_index=item_to_index)
    refactored_data.to_csv(f'./data/{args.d}/{args.d}_final.csv')