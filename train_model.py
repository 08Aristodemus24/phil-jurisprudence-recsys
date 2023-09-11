from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.callbacks import EarlyStopping

from models.test_arcs_a import FM, DFM
from utilities.data_visualizers import view_vars, train_cross_results_v2
from utilities.data_loaders import load_ratings_small, load_raw_movie_ratings_large, load_raw_juris_ratings_large
from utilities.data_preprocessors import normalize_ratings, get_length__build_value_to_index, build_results, split_data
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

    # # load user-item rating and interactions matrices
    # Y, R = load_ratings_small('./data/ratings')
    # n_users, n_items = Y.shape[1], Y.shape[0]
    # view_vars(Y, R, X, THETA, BETA)

    # load user-item rating dataset
    data = dataset[args.d]
    print(data)

    # we must know number of total users and items first before splitting dataset
    # in hate speech classifier we built the word to index dictionary first and
    # configures the embedding matrix to have this dicitonary's number of unique words
    # the embedding look up will have a set number of users & itemstaking into account 
    # the unique users and items in the training, validation, and testing splits
    n_users, user_to_index = get_length__build_value_to_index(data, 'user_id')
    n_items, item_to_index = get_length__build_value_to_index(data, 'item_id')
    
    # modify dataframe column user_id with new indeces from 0 to n_u - 1
    data['user_id'] = data['user_id'].apply(lambda user_id: user_to_index[user_id])
    data['item_id'] = data['item_id'].apply(lambda item_id: item_to_index[item_id])

    # here it is imperative that we split first before normalization
    # to prevent data leakage across the validation and testing sets
    train_data, cross_data, test_data = split_data(data)

    # normalize ratings of each user to an item
    train_data = normalize_ratings(train_data)
    cross_data = normalize_ratings(cross_data)
    test_data = normalize_ratings(test_data)

    # wait what if training set does not contain the users

    

    
    # print(args.n_epochs)

    # model = PhilJurisFM(Y, R, 
    #     num_features=args.n_features, 
    #     epochs=args.n_epochs, 
    #     epoch_to_rec_at=args.epoch_to_rec_at, 
    #     alpha=args.rec_alpha, 
    #     lambda_=args.rec_lambda, 
    #     regularization=args.regularization)
    # history = model.train()

    model = FM(
        n_users=n_users, 
        n_items=n_items,
        emb_dim=args.n_features,
        lambda_=args.rec_lambda,
        regularization=args.regularization)
    
    # model = DFM(
    #     n_users=n_users, 
    #     n_items=n_items, 
    #     emb_dim=args.n_features, 
    #     lambda_=args.rec_lambda, 
    #     keep_prob=args.rec_keep_prob, 
    #     regularization=args.regularization)

    model.compile(
        optimizer=Adam(learning_rate=args.rec_alpha),
        loss=mse_loss(),
        metrics=[mse_metric()]
    )

    history = model.fit(
        [train_data['user_id'], train_data['item_id']],
        train_data['normed_rating'],
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        validation_data=([cross_data['user_id'], cross_data['item_id']], cross_data['normed_rating']),
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    )

    train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='FM (factorization machine) performance')
    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='DFM (deep factorization machine) performance')
    

