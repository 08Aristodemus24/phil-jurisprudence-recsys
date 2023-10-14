import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
import numpy as np


class PhilJurisFM:
    def __init__(self, Y, R, num_features=10, epochs=300, epoch_to_rec_at=50, alpha=0.003, lambda_=0.1, regularization="L2"):
        # this is the user item utility/rating matrix of dimension n_items x n_users
        self._Y = tf.constant(Y)

        # this is the user item interaction matrix of 0s and 1s
        self._R = tf.constant(R)

        # the paramters that need to be optimized
        self._user_embeddings = np.nan
        self._user_embedding_bias = np.nan
        self._item_embeddings = np.nan
        self._item_embedding_bias = np.nan

        # hyper params
        self.epochs = epochs
        self.epoch_to_rec_at = epoch_to_rec_at

        # this num_features is an arbitrary size that the user will pick
        # inj order to decompose the utility/rating matrix Y
        self.num_features = num_features
        self.num_items = Y.shape[0]
        self.num_users = Y.shape[1]
        

        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization

        self.history = {
            'history': {
                'train_loss': [],
                'train_mean_squared_error': [],
                'val_loss': [],
                'val_mean_sqaured_erro': []
            },
            'epochs': []
        }
        self.cost_per_iter = []


if __name__ == "__main__":
    
    # pass initialized tensorflow embedding matrix to embedding lookup and the ids to look up
    user_embeddings = tf.nn.embedding_lookup()
    item_embeddings = tf.nn.embedding_lookup()

    # returns the item and user embeddings that we can use to obtain the dot product
