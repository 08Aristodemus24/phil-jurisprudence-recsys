import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (get_length__build_value_to_index, 
    separate_pos_neg_ratings,
    refactor_raw_ratings,
    split_data,
    normalize_ratings,
    read_item_index_to_entity_id_file, 
    convert_rating, 
    convert_kg
)

from utilities.data_loaders import (load_raw_juris_300k_ratings,
    load_raw_juris_600k_ratings,
    load_raw_movie_1m_ratings, 
    load_raw_movie_20k_kg, 
    load_item_index2entity_id_file)

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError


if __name__ == "__main__":
    # dataset to choose from
    dataset = {
        'juris-300k': load_raw_juris_300k_ratings('./data/juris-300k'),
        'juris-600k': load_raw_juris_600k_ratings('./data/juris-600k'),
        'ml-1m': load_raw_movie_1m_ratings('./data/ml-1m')
    }

    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, default="juris-300k", help='dataset to use which can be juris-300k for the juris docs rating dataset or ml-1m for the movie lens rating dataset')
    parser.add_argument('--protocol', type=str, default="A", help="the protocol or procedure to follow to preprocess the dataset which consists of either preprocessing for binary classification or for regression")
    args = parser.parse_args()

    # load user-item rating dataset
    data = dataset[args.d]
    out_file = args.d.replace('-', '_') + '_ratings'
    # print(data)

    # knowing number of total users and items will be skipped
    # for now just use resulting mapping to preprocess df
    _, user_to_index = get_length__build_value_to_index(data, 'user_id', show_logs=False)
    _, item_to_index = get_length__build_value_to_index(data, 'item_id', show_logs=False)

    # convert old id's to new id's
    data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
    data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

    if args.protocol == "A":
        # separate positive and negative ratings
        pos_user_ratings, neg_user_ratings = separate_pos_neg_ratings(data)

        # finally sample unrated items as our negative class
        refactored_data = refactor_raw_ratings(
            pos_user_ratings=pos_user_ratings, 
            neg_user_ratings=neg_user_ratings, 
            item_to_index=item_to_index, 
            show_logs=False)

        # split data into training, validation, and testing
        train_data, cross_data, test_data = split_data(refactored_data[['user_id', 'item_id']], refactored_data['interaction'])

        # save split data as .csv files
        train_data.to_csv(f'./data/{args.d}/{out_file}_final_train.csv')
        cross_data.to_csv(f'./data/{args.d}/{out_file}_final_cross.csv')
        test_data.to_csv(f'./data/{args.d}/{out_file}_final_test.csv')
    
    elif args.protocol == "B":
        # split data into training, validation, and testing
        # here it is imperative that we split first before normalization
        # to prevent data leakage across the validation and testing sets
        train_data, cross_data, test_data = split_data(data[['user_id', 'item_id']], data['rating'])

        # normalize ratings of each user to an item
        train_data = normalize_ratings(train_data)
        cross_data = normalize_ratings(cross_data)
        test_data = normalize_ratings(test_data)

        train_data.to_csv(f'./data/{args.d}/{out_file}_final_train.csv')
        cross_data.to_csv(f'./data/{args.d}/{out_file}_final_cross.csv')
        test_data.to_csv(f'./data/{args.d}/{out_file}_final_test.csv')

    elif args.protocol == "C":
        # will follow convert rating with knowledge graph into ratings_final.txt or ratings_final.csv
        # convert the knowledge graph and write a kg_final.txt file or kg_final.csv

        # for now this is 
        pass
        