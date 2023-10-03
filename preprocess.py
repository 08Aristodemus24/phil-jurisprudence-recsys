# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (get_unique_values, 
    build_value_to_index,
    column_to_val_to_index,
    separate_pos_neg_ratings,
    refactor_raw_ratings,
    split_data,
    normalize_ratings,
    write_meta_data,
    create_rating_int_matrix,
    build_results,
    read_item_index_to_entity_id_file, 
    convert_rating, 
    convert_kg
)

from utilities.data_loaders import (load_raw_juris_300k_ratings,
    load_raw_juris_600k_ratings,
    load_raw_juris_ratings,
    load_raw_movie_1m_ratings, 
    load_raw_movie_20k_kg, 
    load_item_index2entity_id_file)

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError
import json


def main_preprocess(dataset: str, protocol: str, split_method: str, show_logs: bool):
    """
    preprocesses the data and then returns the training, validation, 
    and testing splits, and returns the unique number of users and items
    in the dataset
    """
    print(f'Commencing preprocessing of {dataset}...')

    # dataset to choose from
    datasets = {
        'juris-300k': load_raw_juris_300k_ratings('./data/juris-300k/juris_300k_ratings.csv'),
        'juris-600k': load_raw_juris_600k_ratings('./data/juris-600k/juris_600k_ratings.csv'),
        'juris-3m': load_raw_juris_ratings('https://raw.githubusercontent.com/08Aristodemus24/LaRJ-Corpus/master/labor%20related%20jurisprudence/juris_2921000_ratings.txt'),
        'ml-1m': load_raw_movie_1m_ratings('./data/ml-1m/ml_1m_ratings.dat')
    }

    data = datasets[dataset]

    # make general file name based on dataset
    out_file = dataset.replace('-', '_')

    if protocol == "A":
        item_to_index = column_to_val_to_index(data, 'item_id')

        # separate positive and negative ratings
        pos_user_ratings, neg_user_ratings = separate_pos_neg_ratings(data, show_logs=False)

        # finally sample unrated items as our negative class
        refactored_data = refactor_raw_ratings(pos_user_ratings, neg_user_ratings, item_to_index, show_logs=False)

        # we must know number of total users and items first before splitting dataset
        # in hate speech classifier we built the word to index dictionary first and
        # configures the embedding matrix to have this dicitonary's number of unique words
        # the embedding look up will have a set number of users & items taking into account 
        # the unique users and items in the training, validation, and testing splits
        user_to_index = build_value_to_index(ratings=refactored_data, column='user_id', show_logs=False)
        n_users = get_unique_values(ratings=refactored_data, column='user_id', show_logs=False)
        item_to_index = build_value_to_index(ratings=refactored_data, column='item_id', show_logs=False)
        n_items = get_unique_values(ratings=refactored_data, column='item_id', show_logs=False)

        # convert old id's to new id's
        refactored_data['user_id'] = refactored_data['user_id'].apply(lambda user_id: user_to_index[user_id])
        refactored_data['item_id'] = refactored_data['item_id'].apply(lambda item_id: item_to_index[item_id])

        # create user-item rating and interaction matrices to check
        # sparsity and inversely density of data
        Y, R = create_rating_int_matrix(data, 'rating', n_users, n_items, show_logs=True)

        # split data into training, validation, and testing
        train_data, cross_data, test_data = split_data(refactored_data[['user_id', 'item_id']], refactored_data['interaction'], option=split_method)

        # define meta data to be used for writing data 
        # for loading to later train model
        meta_data = {
            'n_users': n_users,
            'n_items': n_items
        }

        # write train, cross, and test data so that preprocessing only runs once
        # and outputted files are reusable without having to wait over and over
        train_data.to_csv(f"./data/{dataset}/{out_file}_train.csv")
        cross_data.to_csv(f"./data/{dataset}/{out_file}_cross.csv")
        test_data.to_csv(f"./data/{dataset}/{out_file}_test.csv")

        # write meta data for loading to later train model
        write_meta_data(f'./data/{dataset}/{out_file}_train_meta.json', meta_data)

        if show_logs is True:
            print(f'refactored_data shape: {refactored_data.shape}\n')
            print(f"unique interactions: \n{refactored_data['interaction'].value_counts()}\n")
            print(f"unique user_id's\n{refactored_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{refactored_data['item_id'].value_counts()}\n")

            print(f'train_data shape: {train_data.shape}\n')
            print(f"unique interactions: \n{train_data['interaction'].value_counts()}\n")
            print(f"unique user_id's\n{train_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{train_data['item_id'].value_counts()}\n")

            print(f'cross_data shape: {cross_data.shape}\n')
            print(f"unique interactions: \n{cross_data['interaction'].value_counts()}\n")
            print(f"unique user_id's\n{cross_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{cross_data['item_id'].value_counts()}\n")

            print(f'test_data shape: {test_data.shape}\n')
            print(f"unique interactions: \n{test_data['interaction'].value_counts()}\n")
            print(f"unique user_id's\n{test_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{test_data['item_id'].value_counts()}\n")

            print(f"")
        print('Preprocessing finished!')
    
    elif protocol == "B":
        user_to_index = build_value_to_index(ratings=data, column='user_id', show_logs=False)
        n_users = get_unique_values(ratings=data, column='user_id', show_logs=False)
        item_to_index = build_value_to_index(ratings=data, column='item_id', show_logs=False)
        n_items = get_unique_values(ratings=data, column='item_id', show_logs=False)

        # convert old id's to new id's
        data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
        data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

        # create user-item rating and interaction matrices to check
        # sparsity and inversely density of data
        Y, R = create_rating_int_matrix(data, 'rating', n_users, n_items, show_logs=True)

        # split data into training, validation, and testing
        # here it is imperative that we split first before normalization
        # to prevent data leakage across the validation and testing sets
        train_data, cross_data, test_data = split_data(data[['user_id', 'item_id']], data['rating'])

        # normalize ratings of each user to an item
        train_data = normalize_ratings(train_data)
        cross_data = normalize_ratings(cross_data)
        test_data = normalize_ratings(test_data)

        # define meta data to be used for writing data 
        # for loading to later train model
        meta_data = {
            'n_users': n_users,
            'n_items': n_items
        }

        # write train, cross, and test data so that preprocessing only runs once
        # and outputted files are reusable without having to wait over and over
        train_data.to_csv(f"./data/{dataset}/{out_file}_train.csv")
        cross_data.to_csv(f"./data/{dataset}/{out_file}_cross.csv")
        test_data.to_csv(f"./data/{dataset}/{out_file}_test.csv")

        # write meta data for loading to later train model
        write_meta_data(f'./data/{dataset}/{out_file}_train_meta.json', meta_data)
        if show_logs is True:
            print(f'data shape: {data.shape}\n')
            print(f"unique ratings: \n{data['rating'].value_counts()}\n")
            print(f"unique user_id's\n{data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{data['item_id'].value_counts()}\n")

            print(f'train_data shape: {train_data.shape}\n')
            print(f"unique ratings: \n{train_data['rating'].value_counts()}\n")
            print(f"unique user_id's\n{train_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{train_data['item_id'].value_counts()}\n")

            print(f'cross_data shape: {cross_data.shape}\n')
            print(f"unique ratings: \n{cross_data['rating'].value_counts()}\n")
            print(f"unique user_id's\n{cross_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{cross_data['item_id'].value_counts()}\n")

            print(f'test_data shape: {test_data.shape}\n')
            print(f"unique ratings: \n{test_data['rating'].value_counts()}\n")
            print(f"unique user_id's\n{test_data['user_id'].value_counts()}\n")
            print(f"unique item_id's\n{test_data['item_id'].value_counts()}\n")
        print('Preprocessing finished!')

    elif protocol == "C":
        # will follow convert rating with knowledge graph into ratings_final.txt or ratings_final.csv
        # convert the knowledge graph and write a kg_final.txt file or kg_final.csv

        # for now this is 
        pass

if __name__ == "__main__":
    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, default="juris-300k", help='dataset to use which can be juris-300k for the juris docs rating dataset or ml-1m for the movie lens rating dataset')
    parser.add_argument('--protocol', type=str, default="A", help="the protocol or procedure to follow to preprocess the dataset which consists of either preprocessing for binary classification or for regression")
    parser.add_argument('--split_method', type=str, default="intact", help="the method to use to split the dataset which either \
        preserves the order of the unique users in training data based on the final preprocessed dataset even after splitting or \
        just randomly splits the data and shuffles it")
    parser.add_argument('--show_logs', type=bool, default=True, help='shows logs to view important values after preprocessing')
    args = parser.parse_args()

    main_preprocess(args.d, args.protocol, args.split_method, show_logs=args.show_logs)