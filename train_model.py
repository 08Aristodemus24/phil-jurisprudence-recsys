from models.model_arcs import PhilJurisFM, FM, DFM
from utilities.data_visualizers import view_vars, train_cross_results_v2
from utilities.data_loaders import load_ratings_small, load_precalc_params_small, load_Movie_List_pd, load_raw_ratings_large
from argparse import ArgumentParser, ArgumentTypeError, ArgumentError


if __name__ == "__main__":
    # # load user-item rating and interactions matrices
    # Y, R = load_ratings_small('./data/ratings')
    # n_users, n_items = Y.shape[1], Y.shape[0]

    # # # view data
    # # view_vars(Y, R, X, THETA, BETA)

    # # instantiate parser to take args from user in command line
    # parser = ArgumentParser()
    # parser.add_argument('--n_features', type=int, default=10, help='number of features of decomposed matrices X, THETA, and B of Y')
    # parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')
    # parser.add_argument('--epoch_to_rec_at', type=int, default=50, help='every epoch to record at')
    # parser.add_argument('--rec_alpha', type=float, default=0.003, help='learning rate of recommendation task')
    # parser.add_argument('--rec_lambda', type=float, default=0.1, help='lambda value of regularization term in recommendation task')
    # parser.add_argument('--regularization', type=str, default="L2", help='regularizer to use in regularization term')
    # # parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    # # parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
    # # parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
    # args = parser.parse_args()
    # # print(args.n_epochs)

    # model = PhilJurisFM(Y, R, 
    #     num_features=args.n_features, 
    #     epochs=args.n_epochs, 
    #     epoch_to_rec_at=args.epoch_to_rec_at, 
    #     alpha=args.rec_alpha, 
    #     lambda_=args.rec_lambda, 
    #     regularization=args.regularization)
    # history = model.train()

    # train_cross_results_v2(results=history['history'], epochs=history['epochs'], img_title='train loss of collaborative filtering SVD across epochs')

    # load user-item rating dataset
    data = load_raw_ratings_large('./data/ml-1m')
    print(data)
    
    # model = FM()
    
    
    

