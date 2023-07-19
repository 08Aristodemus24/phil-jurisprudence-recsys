import tensorflow as tf
from tensorflow import GradientTape
import numpy as np

class PhilJurisCollabFilter:
    def __init__(self, Y, R, epochs=300, alpha=0.003):
        # this is the user item utility matrix
        self.Y = Y

        # this is the user item interaction matrix of 0s and 1s
        self.R = R

        # the paramters that need to be optimized
        self.THETA = 0
        self.BETA = 0
        self.X = 0

        # hyper params
        self.epochs = epochs
        self.alpha = alpha

        self.num_instances = 0
        self.num_features = 0
        self.num_users = 0
        self.num_items = 0

    def train(self):
        pass

    def init_mini_batches(self, mini_batch_size=32):
        """
        X here will have a shape of num_features x num_instances
        this is why we use our permutated indeces in the 2nd dim
        when indexing the numpy array to obtain our shuffled input X's
        args:
            
        """
        m = self.num_instances

        mini_batches = []

        # say we had 50 examples, permutation(50) would give a 
        # permutated numpy array with values 1 to 50
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]


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




        

        