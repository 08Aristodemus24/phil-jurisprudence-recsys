import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from models.test_arcs_d import FM, DFM
import numpy as np
import math

from utilities.data_loaders import load_data_splits
from utilities.data_preprocessors import (build_results,
    load_meta_data)

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError



def generate_batches(X, Y, batch_size=32):
    # number of examples in data split
    m = X.shape[0]

    # number of increments for index 
    inc = batch_size

    # shuffle data
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]

    # number of complete mini batches
    num_comp_batches = math.floor(m / batch_size)

    # yield batches
    for k in range(0, num_comp_batches):
        # obtain slice/batch of the dataset
        X_batch_k = shuffled_X[k * inc: batch_size * (k + 1), :]
        Y_batch_k = shuffled_Y[k * inc: batch_size * (k + 1)]

        # yield a complete batch of size batch_size
        yield (X_batch_k, Y_batch_k)

    if (m % batch_size) != 0:
        X_batch_last = shuffled_X[batch_size * (k + 1):, :]
        Y_batch_last = shuffled_Y[batch_size * (k + 1):]
        
        # yield the last uncomplete batch of the dataset
        # shoudl number of samples divided by batch size result in 
        # remainder thus having our last mini batch be incomplete
        yield (X_batch_last, Y_batch_last)



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
    args = parser.parse_args()

    # make general file name based on dataset
    out_file = args.d.replace('-', '_')

    # load user-item rating data splits and meta data
    meta_data = load_meta_data(f'./data/{args.d}/{out_file}_train_meta.json')
    n_users, n_items = meta_data['n_users'], meta_data['n_items']
    train_data, cross_data, test_data = load_data_splits(args.d, f'./data/{args.d}')

    train_data = generate_batches(train_data[['user_id', 'item_id']].to_numpy(), train_data['normed_rating'].to_numpy(), batch_size=args.batch_size)
    cross_data = generate_batches(cross_data[['user_id', 'item_id']].to_numpy(), cross_data['normed_rating'].to_numpy(), batch_size=args.batch_size)
    test_data = generate_batches(test_data[['user_id', 'item_id']].to_numpy(), test_data['normed_rating'].to_numpy(), batch_size=args.batch_size)

    # define loss function
    loss_fn = MeanSquaredError()

    # define optimizer
    optimizer = Adam(learning_rate=args.rec_alpha)

    # instantiate the model with the necessary hyperparams
    # and important meta data
    model = FM(n_users, n_items)

    train_rmse_metric = RootMeanSquaredError()
    cross_rmse_metric = RootMeanSquaredError()

    
    for epoch in range(args.n_epochs):
        # training loop
        for batch_idx, (X_train, Y_train) in enumerate(train_data):
            print(X_train)
            with GradientTape() as tape:
                # pass the series of ids to the model to predict the y
                Y_pred = model(X_train, training=True)

                # pass the prediction to the loss function
                train_loss = loss_fn(Y_train, Y_pred)

                # gradient descent
                grads = tape.gradient(train_loss, model.trainable_weights)

            # update parameters/coefficients/embeddings
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_rmse_metric.update_state(Y_train, Y_pred)

        train_rmse = train_rmse_metric.result()
        print(f'mean_squared_error at epoch {epoch}: {train_loss}')
        print(f'rmse at epoch {epoch}: {train_rmse}')


# # testing loop
# for batch_idx, (X_cross, Y_cross) in enumerate(cross_data):
#     # get the predicted Y's using the validation data set
#     Y_pred = model(X_cross, training=False)

#     # compute the loss for the validation data set
#     cross_loss = loss_fn(Y_cross, Y_pred)

#     cross_rmse_metric.update_state(Y_cross, Y_pred)

# cross_rmse = cross_rmse_metric.result()
# print(f'cross mse at epoch {epoch}: {cross_loss}')
# print(f'cross rmse at epoch {epoch}: {cross_rmse}')

# train_rmse_metric.reset_states()
# cross_rmse_metric.reset_states()

        