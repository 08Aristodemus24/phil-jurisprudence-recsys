from tensorflow.keras.callbacks import EarlyStopping
from preprocess import main_preprocess
from models.model_loader import load_model

# the functions from the preprocess.py file which we need to use in order to get hte
# intermediate values before ratings_final.txt and kg_final.txt are outputted
from utilities.data_preprocessors import (build_results,
    load_meta_data,
    read_item_index_to_entity_id_file, 
    convert_rating, 
    convert_kg
)

from utilities.data_visualizers import view_vars, train_cross_results_v2
from utilities.data_loaders import load_data_splits

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError



if __name__ == "__main__":
    # instantiate parser to take args from user in command line
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, default="juris-300k", help='dataset to use which can be juris-300k for the juris docs rating dataset or ml-1m for the movie lens rating dataset')
    parser.add_argument('--protocol', type=str, default="A", help="the protocol or procedure to follow to preprocess the dataset which consists of either preprocessing for binary classification or for regression")
    parser.add_argument('--model_name', type=str, default="FM", help="which specific model to train")
    parser.add_argument('--n_features', type=int, default=10, help='number of features of decomposed matrices X, THETA, B_u, and B_i of Y')
    parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('--layers_dims', nargs='+', type=int, default=[16, 16, 16], help='number of layers and number of nodes in each layer of the dnn')
    parser.add_argument('--epoch_to_rec_at', type=int, default=50, help='every epoch to record at')
    parser.add_argument('--rec_alpha', type=float, default=1e-4, help='learning rate of recommendation task')
    parser.add_argument('--rec_lambda', type=float, default=0.1, help='lambda value of regularization term in recommendation task')
    parser.add_argument('--rec_keep_prob', type=float, default=1, help='lambda value of regularization term in recommendation task')
    parser.add_argument('--regularization', type=str, default="L2", help='regularizer to use in regularization term')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    # parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
    # parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
    args = parser.parse_args()

    # make general file name based on dataset
    out_file = args.d.replace('-', '_')

    # load user-item rating data splits and meta data
    meta_data = load_meta_data(f'./data/{args.d}/{out_file}_train_meta.json')
    n_users, n_items = meta_data['n_users'], meta_data['n_items']
    train_data, cross_data, test_data = load_data_splits(args.d, f'./data/{args.d}')

    # load model
    model = load_model(
        model_name=args.model_name, 
        protocol=args.protocol,
        n_users=n_users,
        n_items=n_items,
        n_features=args.n_features,
        layers_dims=args.layers_dims,
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
        callbacks=[EarlyStopping(monitor='val_f1_m', patience=50)]
    )

    # visualize model results
    # train_cross_results_v2(
    #     results=build_results(history, metrics=['binary_crossentropy', 'val_binary_crossentropy', 'f1_m', 'val_f1_m', 'auc', 'val_auc']), 
    #     epochs=history.epoch, 
    #     img_title='binary FM (factorization machine) performance')
    
    train_cross_results_v2(
        results=build_results(history, metrics=['binary_crossentropy', 'val_binary_crossentropy', 'f1_m', 'val_f1_m', 'auc', 'val_auc']), 
        epochs=history.epoch, 
        img_title='binary DFM (deep factorization machine) performance', 
        image_only=True)

    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='FM (factorization machine) performance')
    # train_cross_results_v2(results=build_results(history, metrics=['loss', 'val_loss',]), epochs=history.epoch, img_title='DFM (deep factorization machine) performance')