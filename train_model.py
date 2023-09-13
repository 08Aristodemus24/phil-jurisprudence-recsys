from keras import backend as K
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

from tensorflow.keras.callbacks import EarlyStopping

# from models.test_arcs_a import FM, DFM, MKR
from models.model_arcs import FM, DFM, MKR
from utilities.data_visualizers import view_vars, train_cross_results_v2

# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (get_length__build_value_to_index, 
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

    # we must know number of total users and items first before splitting dataset
    # in hate speech classifier we built the word to index dictionary first and
    # configures the embedding matrix to have this dicitonary's number of unique words
    # the embedding look up will have a set number of users & items taking into account 
    # the unique users and items in the training, validation, and testing splits
    n_users, user_to_index = get_length__build_value_to_index(data, 'user_id', show_logs=False)
    n_items, item_to_index = get_length__build_value_to_index(data, 'item_id', show_logs=False)

    # convert old id's to new id's
    data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
    data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

    if protocol == "A":
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
        # return refactored_data['user_id'].value_counts().size, refactored_data['item_id'].value_counts().size, train_data, cross_data, test_data
    
    elif protocol == "B":
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
            'metrics': [bce_metric(), BinaryAccuracy(), Precision(), Recall(), AUC(), f1_m]
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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



if __name__ == "__main__":
    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, default="juris-300k", help='dataset to use which can be juris-300k for the juris docs rating dataset or ml-1m for the movie lens rating dataset')
    parser.add_argument('--protocol', type=str, default="A", help="the protocol or procedure to follow to preprocess the dataset which consists of either preprocessing for binary classification or for regression")
    parser.add_argument('--model_name', type=str, default="FM", help="which specific model to train")
    parser.add_argument('--n_features', type=int, default=10, help='number of features of decomposed matrices X, THETA, B_u, and B_i of Y')
    parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('--epoch_to_rec_at', type=int, default=50, help='every epoch to record at')
    parser.add_argument('--rec_alpha', type=float, default=1e-4, help='learning rate of recommendation task')
    parser.add_argument('--rec_lambda', type=float, default=0.1, help='lambda value of regularization term in recommendation task')
    parser.add_argument('--rec_keep_prob', type=float, default=1, help='lambda value of regularization term in recommendation task')
    parser.add_argument('--regularization', type=str, default="L2", help='regularizer to use in regularization term')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    # parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
    # parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
    args = parser.parse_args()
    # print(args)

    # load user-item rating dataset
    n_users, n_items, train_data, cross_data, test_data = main_preprocess(args.d, args.protocol, show_logs=False)

    # load model
    model = load_model(
        model_name=args.model_name, 
        protocol=args.protocol,
        n_users=n_users,
        n_items=n_items,
        n_features=args.n_features,
        epoch_to_rec_at=args.epoch_to_rec_at,
        rec_alpha=args.rec_alpha,
        rec_lambda=args.rec_lambda,
        rec_keep_prob=args.rec_keep_prob,
        regularization=args.regularization
    )

    # train model
    history = model.fit(
        [train_data['user_id'], train_data['item_id']],
        train_data['interaction'],
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        validation_data=([cross_data['user_id'], cross_data['item_id']], cross_data['interaction']),
        # callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    )

    # # visualize model results
    # train_cross_results_v2(
    #     results=build_results(history, metrics=['binary_crossentropy', 'val_binary_crossentropy', 'f1_m', 'val_f1_m', 'auc', 'val_auc']), 
    #     epochs=history.epoch, 
    #     img_title='binary FM (factorization machine) performance')
    
    # train_cross_results_v2(
    #     results=build_results(history, metrics=['binary_crossentropy', 'val_binary_crossentropy', 'f1_m', 'val_f1_m', 'auc', 'val_auc']), 
    #     epochs=history.epoch, 
    #     img_title='binary DFM (deep factorization machine) performance')

    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='FM (factorization machine) performance')
    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='DFM (deep factorization machine) performance')