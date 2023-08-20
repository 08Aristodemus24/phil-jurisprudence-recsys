from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.callbacks import EarlyStopping

from models.model_arcs import PhilJurisFM, FM, DFM
from utilities.data_visualizers import view_vars, train_cross_results_v2
from utilities.data_loaders import load_ratings_small, load_raw_ratings_large
from utilities.data_preprocessors import normalize_ratings, get_length__build_value_to_index, build_results
from argparse import ArgumentParser, ArgumentTypeError, ArgumentError



if __name__ == "__main__":
    # # load user-item rating and interactions matrices
    # Y, R = load_ratings_small('./data/ratings')
    # n_users, n_items = Y.shape[1], Y.shape[0]
    # view_vars(Y, R, X, THETA, BETA)

    # load user-item rating dataset
    ml_1m_ratings = load_raw_ratings_large('./data/ml-1m')
    n_users, user_to_index = get_length__build_value_to_index(ml_1m_ratings, 'user_id')
    n_items, item_to_index = get_length__build_value_to_index(ml_1m_ratings, 'item_id')

    # normalize ratings of each user to an item
    ml_1m_ratings = normalize_ratings(ml_1m_ratings)

    # modify dataframe column user_id with new indeces from 0 to n_u - 1
    ml_1m_ratings['user_id'] = ml_1m_ratings['user_id'].apply(lambda user_id: user_to_index[user_id])
    ml_1m_ratings['item_id'] = ml_1m_ratings['item_id'].apply(lambda item_id: item_to_index[item_id])

    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
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
    # print(args.n_epochs)

    # model = PhilJurisFM(Y, R, 
    #     num_features=args.n_features, 
    #     epochs=args.n_epochs, 
    #     epoch_to_rec_at=args.epoch_to_rec_at, 
    #     alpha=args.rec_alpha, 
    #     lambda_=args.rec_lambda, 
    #     regularization=args.regularization)
    # history = model.train()

    # model = FM(n_users=n_users, n_items=n_items, emb_dim=args.n_features, lambda_=args.rec_lambda, regularization=args.regularization)
    
    model = DFM(
        n_users=n_users, 
        n_items=n_items, 
        emb_dim=args.n_features, 
        lambda_=args.rec_lambda, 
        keep_prob=args.rec_keep_prob, 
        regularization=args.regularization)

    model.compile(
        optimizer=Adam(learning_rate=args.rec_alpha),
        loss=mse_loss(),
        metrics=[mse_metric()]
    )

    history = model.fit(
        [ml_1m_ratings['user_id'], ml_1m_ratings['item_id']],
        ml_1m_ratings['normed_rating'],
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        validation_split=0.3,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    )

    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='FM (factorization machine) performance')
    train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='DFM (deep factorization machine) performance')
    

