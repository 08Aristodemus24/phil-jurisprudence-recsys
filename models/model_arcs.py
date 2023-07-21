import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
import numpy as np

class PhilJurisCollabFilter:
    def __init__(self, Y, R, num_features, epochs=300, epoch_to_rec_at=50, alpha=0.003, lambda_=0.1, regularization="L2"):
        # this is the user item utility/rating matrix of dimension n_items x n_users
        self._Y = tf.constant(Y)

        # this is the user item interaction matrix of 0s and 1s
        self._R = tf.constant(R)

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

    def train(self, show_vars=True):
        # init params here
        self.init_params()
    
        # set utility matrix Y and interaction matrix R to tf constants
        Y, R = self.Y, self.R

        # get newly initialized params
        X, THETA, BETA = self.X, self.THETA, self.BETA

        # instantiate optimizer
        optimizer = Adam(learning_rate=self.alpha)

        for epoch in range(self.epochs):
            with GradientTape as tape:
                # multiply resulting dot products to 
                # user-item interaction matrix R
                logits = self.linear(X, THETA, BETA)

                # calculate the cost
                cost = self.J_mean_squared(logits, Y, R) + self.regularizer(X, THETA)

            # take derivative of cost with respect to params X, THETA, and BETA
            grads = tape.gradient(cost, [X, THETA, BETA])

            # apply gradients to params X, THETA, and BETA
            optimizer.apply_gradients(zip(grads, [X, THETA, BETA]))

            

            if ((epoch % self.rec_ep_at) == 0) or (epoch == self.epochs - 1):
                # record all previous values after applying gradients
                self.history['epoch'].append(epoch)
                self.history['history']['train_loss'].append(cost)
                # self.history['history']['train_categorical_accuracy'].append(train_acc)
                # self.history['history']['val_loss'].append(val_cost)
                # self.history['history']['val_categorical_accuracy'].append(val_acc)

                print(f"epoch {epoch} - train_loss: {cost}")
                # print(f"epoch {epoch} - train_loss: {train_cost} - train_categorical_accuracy: {'{:.2%}'.format(train_acc)} - val_loss: {val_cost} - val_categorical_accuracy: {'{:.2%}'.format(val_acc)}")


                if show_vars == True:
                    pass

        # set new coefficients after training
        self.THETA = THETA
        self.BETA = BETA

        return self.history

        # start trainign loop here
            # do forward pass
            # pass result to cost function
            # calculate regularizer and add to cost
            # extract gradients
            # apply gradients
            # records loss

    def init_params(self):
        # extract number of users, items, and given arbitrary number of features by user
        n_users, n_items, n_features = self.num_users, self.num_items, self.num_features

        # initialize random values of parameters to optimize from normal distribution
        self.X = tf.Variable(tf.random.normal(shape=(n_items, n_features), dtype=tf.float64, name='X'))
        self.THETA = tf.Variable(tf.random.normal(shape=(n_users, n_features), dtype=tf.float64, name='THETA'))
        self.BETA = tf.Variable(tf.random.normal(shape=(1, n_users), dtype=tf.float64, name='BETA'))

    def linear(self, X, THETA, BETA):
        """
            args:
                X - is the feature matrix of (n_items x n_features) dimensionality
                THETA - is the non-bias coefficients of (n_users x n_features) dimensionality
                BETA - is the bias coefficients (1 x n_users) dimensionality
        """

        return tf.linalg.matmul(X, tf.transpose(THETA)) + BETA

    @property
    def Y(self):
        return self._Y
    
    @property
    def R(self):
        return self._R

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

    def regularizer(self, X, THETA):
        if self.regularization.upper() == "L2":
            pass
            # # get the square of all coefficients in each layer excluding biases
            # # if 5 layers then loop from 0 to 3 to access all coefficients
            # sum_sq_coeffs = tf.reduce_sum(tf.math.square(THETA))

            # # multiply by lambda constant then divide by 2m
            # l2_norm = (self.lambda_ * sum_sq_coeffs) / (2 * self.num_instances)

            # # return l2 norm
            # return l2_norm

        elif self.regularization.upper() == "L1":
            pass
            # # if there is only 2 layers then calculation
            # # in loop only runs once
            # sum_abs_coeffs = tf.reduce_sum(tf.math.abs(THETA))

            # # multiply by lambda constant then divide by 2m
            # l1_norm = (self.lambda_ * sum_abs_coeffs) / (2 * self.num_instances)

            # # return l1 norm
            # return l1_norm
        
        # return 0 if no regularizer is indicated
        return 0

    def J_mean_squared(self, A, Y, R):
        """
            args:
                A - the predicted rating matrix
                Y - the real user-item rating matrix
                R - the user-item interaction matrix to use as multiplier to difference of
                A and Y to make sure only items rated by users is kept because of the ones
                in this matrix
        """

        cost = tf.reduce_sum(tf.math.square(A - Y) * R) / 2
        return cost 

    # add here initialization of parameters THETA, BETA, and X




        

        