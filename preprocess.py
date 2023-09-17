from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (BinaryCrossentropy as bce_loss, 
    MeanSquaredError as mse_loss
)

from tensorflow.keras.metrics import (BinaryAccuracy, 
    Precision,
    Recall,
    AUC,
    BinaryCrossentropy as bce_metric, 
    MeanSquaredError as mse_metric
)

# from models.test_arcs_a import FM, DFM, MKR
from models.model_arcs import FM, DFM, MKR
# from metrics.custom_metrics import f1_m

# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (get_unique_values, 
    build_value_to_index,
    column_to_val_to_index,
    separate_pos_neg_ratings,
    refactor_raw_ratings,
    split_data,
    normalize_ratings,
    build_results,
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
import json


def main_preprocess(dataset: str, protocol: str, show_logs: bool=True):
    """
    preprocesses the data and then returns the training, validation, 
    and testing splits, and returns the unique number of users and items
    in the dataset
    """
    print(f'Commencing preprocessing of {dataset}...')

    # dataset to choose from
    datasets = {
        'juris-300k': load_raw_juris_300k_ratings('./data/juris-300k'),
        'juris-600k': load_raw_juris_600k_ratings('./data/juris-600k'),
        'ml-1m': load_raw_movie_1m_ratings('./data/ml-1m')
    }

    data = datasets[dataset]

    if protocol == "A":
        item_to_index = column_to_val_to_index(data, 'item_id')

        # separate positive and negative ratings
        pos_user_ratings, neg_user_ratings = separate_pos_neg_ratings(data, show_logs=True)

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

        # split data into training, validation, and testing
        train_data, cross_data, test_data = split_data(refactored_data[['user_id', 'item_id']], refactored_data['interaction'])

        if show_logs is True:
            print(f"unique interactions: {refactored_data['interaction'].value_counts()}")
            print(f"unique user_id's{refactored_data['user_id'].value_counts()}")
            print(f"unique item_id's{refactored_data['item_id'].value_counts()}")
            print(f'train_data shape: {train_data.shape}')
            print(f'cross_data shape: {cross_data.shape}')
            print(f'test_data shape: {test_data.shape}')
        print('Preprocessing finished!')

        # return data splits
        return n_users, n_items, train_data, cross_data, test_data
    
    elif protocol == "B":
        user_to_index = build_value_to_index(ratings=data, column='user_id', show_logs=False)
        n_users = get_unique_values(ratings=data, column='user_id', show_logs=False)
        item_to_index = build_value_to_index(ratings=data, column='item_id', show_logs=False)
        n_items = get_unique_values(ratings=data, column='item_id', show_logs=False)

        # convert old id's to new id's
        data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
        data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

        # split data into training, validation, and testing
        # here it is imperative that we split first before normalization
        # to prevent data leakage across the validation and testing sets
        train_data, cross_data, test_data = split_data(data[['user_id', 'item_id']], data['rating'])

        # normalize ratings of each user to an item
        train_data = normalize_ratings(train_data)
        cross_data = normalize_ratings(cross_data)
        test_data = normalize_ratings(test_data)
        print('Preprocessing finished!')

        # return data splits
        return n_users, n_items, train_data, cross_data, test_data

    elif protocol == "C":
        # will follow convert rating with knowledge graph into ratings_final.txt or ratings_final.csv
        # convert the knowledge graph and write a kg_final.txt file or kg_final.csv

        # for now this is 
        pass

def load_model(model_name: str, protocol: str, n_users: int, n_items: int, n_features: int, epoch_to_rec_at: int, rec_alpha: float, rec_lambda: float, rec_keep_prob: float, regularization: str):
    """
    creates, compiles and returns chosen model to train

    args: 
        model_name - 
        protocol - 
        n_users - 
        n_items - 
        n_features - 
        epoch_to_rec_at - 
        rec_alpha - 
        rec_lambda - 
        rec_keep_prob - 
        regularization - 
    """
    
    protocols = {
        'A': {
            'loss': bce_loss(),
            'metrics': [bce_metric(), BinaryAccuracy(), Precision(), Recall(), AUC()]#.extend([f1_m])
        },
        'B': {
            'loss': mse_loss(),
            'metrics': [mse_metric()]
        }
    }

    models = {
        'FM': FM(
            n_users=n_users, 
            n_items=n_items,
            emb_dim=n_features,
            lambda_=rec_lambda,
            regularization=regularization),
        'DFM': DFM(
            n_users=n_users, 
            n_items=n_items, 
            emb_dim=n_features,
            layers_dims=[8],
            lambda_=rec_lambda, 
            keep_prob=rec_keep_prob, 
            regularization=regularization)
    }

    model = models[model_name]

    model.compile(
        optimizer=Adam(learning_rate=rec_alpha),
        loss=protocols[protocol]['loss'],
        metrics=protocols[protocol]['metrics']
    )
       
    return model