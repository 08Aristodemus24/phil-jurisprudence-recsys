import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import math

from utilities.data_loaders import load_data_splits
from utilities.data_preprocessors import (build_results,
    load_meta_data)

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError


tf.compat.v1.disable_eager_execution()

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

    # initialize user and item embeddings
    # initialize random values of parameters to optimize from normal distribution
    item_emb_matrix = tf.Variable(tf.random.normal(shape=(n_items, args.n_features), dtype=tf.float64))
    user_emb_matrix = tf.Variable(tf.random.normal(shape=(n_users, args.n_features), dtype=tf.float64))
    user_emb_bias_vec = tf.Variable(tf.random.normal(shape=(n_users, 1), dtype=tf.float64))
    item_emb_bias_vec = tf.Variable(tf.random.normal(shape=(n_items, 1), dtype=tf.float64))

    # define loss function
    loss_fn = MeanSquaredError()

    # define optimizer
    optimizer = Adam(learning_rate=args.rec_alpha)

    for epoch in range(args.n_epochs):
        for batch_idx, (X_train, Y_train) in enumerate(train_data):
            # expand dimesnions of Y_train to m x 1 
            # since it is currently just (m, )
            Y_train = tf.expand_dims(Y_train, axis=1)

            # get number of trainign examples
            m = X_train.shape[0]

            # retrieve user and item embeddings, representing the 
            # 0th and 1st columns respectively since we did convert
            # the dataset to a numpy array object
            user_embeddings = tf.nn.embedding_lookup(user_emb_matrix, X_train[:, 0])
            item_embeddings = tf.nn.embedding_lookup(item_emb_matrix, X_train[:, 1])
            user_embedding_bias = tf.nn.embedding_lookup(user_emb_bias_vec, X_train[:, 0])
            item_embedding_bias = tf.nn.embedding_lookup(item_emb_bias_vec, X_train[:, 1])

            user_embeddings = tf.expand_dims(user_embeddings, axis=1)
            item_embeddings = tf.expand_dims(item_embeddings, axis=1)

            with GradientTape() as tape:
                # perform the linear operation
                dot = tf.linalg.matmul(user_embeddings, tf.transpose(item_embeddings, perm=[0, 2, 1]))
                Y_pred = tf.reshape(dot, shape=(m, -1)) + user_embedding_bias + item_embedding_bias
                print(Y_pred)
                print(Y_train.shape)
                

                # pass the prediction to the loss function
                loss = loss_fn(Y_train, Y_pred)
                print(loss)

            # gradient descent
            grads = tape.gradient(loss, [user_embeddings, item_embeddings, user_embedding_bias, item_embedding_bias])

            # update parameters/coefficients/embeddings
            optimizer.apply_gradients(zip(grads, [user_embeddings, item_embeddings, user_embedding_bias, item_embedding_bias]))

        print(f'mean_squared_error at epoch {epoch}: {loss}')




                
    
