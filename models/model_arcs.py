import tensorflow as tf
from tensorflow import GradientTape
import numpy as np

class PhilJurisCollabFilter:
    def __init__(self, Y, R, num_features, epochs=300, epoch_to_rec_at=50, alpha=0.003, lambda_=0.1, regularization="L2"):
        # this is the user item utility/rating matrix of dimension n_items x n_users
        self._Y = Y

        # this is the user item interaction matrix of 0s and 1s
        self._R = R

        # the paramters that need to be optimized
        self._THETA = np.nan
        self._BETA = np.nan
        self._X = np.nan

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
                'train_categorical_accuracy': [],
                'val_loss': [],
                'val_categorical_accuracy': []
            },
            'epoch': []
        }
        self.cost_per_iter = []

    def train(self):
        pass

    def init_params(self):
        # extract number of users, items, and given arbitrary number of features by user
        n_users, n_items, n_features = self.num_users, self.num_items, self.num_features

        # initialize random values of parameters to optimize from normal distribution
        self.X = tf.Variable(tf.random.normal(shape=(n_items, n_features), dtype=tf.float64, name='X'))
        self.THETA = tf.Variable(tf.random.normal(shape=(n_users, n_features), dtype=tf.float64, name='THETA'))
        self.BETA = tf.Variable(tf.random.normal(shape=(1, n_users), dtype=tf.float64, name='BETA'))

    @property
    def Y(self):
        return self._Y
    
    @property
    def R(self):
        return

    @property
    def X(self):
        return self._X

    @property
    def THETA(self):
        return self._THETA
    
    @property
    def BETA(self):
        return self._BETA
    
    @X.setter
    def X(self, new_val):
        self._X = new_val

    @THETA.setter
    def THETA(self, new_val):
        self._THETA = new_val
    
    @BETA.setter
    def BETA(self, new_val):
        self._BETA = new_val
    
    def normalizeRatings(Y, R):
        """
        Preprocess data by subtracting mean rating for every movie (every row).
        Only include real ratings R(i,j)=1.
        [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
        has a rating of 0 on average. Unrated moves then have a mean rating (0)
        Returns the mean rating in Ymean.
        """
        Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
        Ynorm = Y - np.multiply(Ymean, R) 
        return(Ynorm, Ymean)

    def predict():
        pass

    def restore_mean():
        pass

    def J_mean_squared(self, A, Y):
        pass

    # add here initialization of parameters THETA, BETA, and X




        

        