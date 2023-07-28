import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2, L1
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow import GradientTape

import numpy as np



class FM(tf.keras.Model):
    def __init__(self, n_users, n_items, emb_dim=32, lambda_=1, regularization="L2"):
        """
        Implements the Factorization Machine (FM) architecture by subclassing
        built-in Model
        """
        super().__init__()
        # number of unique users and items
        self.n_users = n_users
        self.n_items = n_items

        # hyperparams
        self.lambda_ = lambda_
        self.emb_dim = emb_dim

        # regularization to use
        self.regularization = L2 if regularization == "L2" else L1

        # initialize embedding and embedding bias layers
        self.user_emb_layer = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_regularizer=self.regularization(lambda_))
        self.item_emb_layer = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_regularizer=self.regularization(lambda_))

        self.user_emb_bias_layer = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros')
        self.item_emb_bias_layer = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer='zeros')

        # initialize output layer
        self.dot_layer = tf.keras.layers.Dot(axes=(2, 1))
        self.add_layer = tf.keras.layers.Add()


    def call(self, inputs, training=False):
        # catch inputs first since Model will be taking in a 2 rows of data
        # the user_id_input which is m x 1 and item_id_input which is m x 1
        # since one example would be one user and one item
        user_id_input = inputs[0]
        item_id_input = inputs[1]

        # define forward pass            
        user_emb = self.user_emb_layer(user_id_input)
        item_emb = self.item_emb_layer(item_id_input)

        user_emb_bias = self.user_emb_bias_layer(user_id_input)
        item_emb_bias = self.item_emb_bias_layer(item_id_input)

        # since it is a mere FM model only a linear calculation will be sufficient
        # which is the dot product of the two user_embedding and item_embedding vectors 
        # plus the user_bias and item_bias scalars
        # out = tf.linalg.matmul(user_emb, tf.transpose(item_emb, perm=[0, 2, 1])) + user_emb_bias + item_emb_bias
        user_item_dot = self.dot_layer([user_emb, tf.transpose(item_emb, perm=[0, 2, 1])])
        out = self.add_layer([user_item_dot, user_emb_bias, item_emb_bias])

        return out



class DFM(tf.keras.Model):
    def __init__(self, n_users, n_items, emb_dim=32, layers_dims=[16, 16, 16], lambda_=1e-6):
        """
        
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        # hyperparams
        self.lambda_ = lambda_

        self.emb_dim = emb_dim
        self.layers_dims = layers_dims
        self.num_layers = len(layers_dims)

        # seems to have no input shape so invetigate this
        # initialize input layers
        self.user_id_input = tf.keras.Input(shape=(), name='user_id', dtype=tf.int64)
        self.item_id_input = tf.keras.Input(shape=(), name='item_id', dtype=tf.int64)

        # initialize embedding and embedding bias layers
        self.user_emb_layer = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_regularizer=L2(lambda_))
        self.item_emb_layer = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_regularizer=L2(lambda_))

        self.user_emb_bias_layer = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros')
        self.item_emb_bias_layer = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer='zeros')

        # initialize dense and activation layers
        self.dense_layers, self.act_layers = self.init_dense_act_layers()

        # A last
        self.last_a_layer = tf.keras.layers.Dense(units=1, kernel_regularizer=L2(lambda_))

        # output
        self.output_layer = tf.keras.layers.Activation(activation=tf.nn.sigmoid)


    def call(self, inputs, training=False):
        """
        
        """
        # will perform forward propagation
        user_embedding = self.user_emb_layer(self.user_id_input)
        item_embedding = self.item_emb_layer(self.item_id_input)

        user_bias = self.user_emb_bias_layer(self.user_id_input)
        item_bias = self.item_emb_bias_layer(self.item_id_input)

        fact_matrix = tf.reduce_sum(user_embedding * item_embedding, axis=1, keepdims=True) + user_bias + item_bias

        # this will be A0 or the first input layer of this deep neural network
        A = tf.concat([user_embedding, item_embedding], axis=1)

        # forward propagate through deep neural network according to number of layers
        for l in range(self.num_layers):
            # pass concatenated user_embedding and item embedding 
            # to dense layer to calculate Z at layer l
            Z = self.dense_layers[l](A)

            # activate output Z layer by passing to relu activation layer
            A = self.act_layers[l](Z)

        # one linear layer
        A_last = self.last_a_layer(A)

        OUT = self.output_layer(fact_matrix + A)

        return OUT
    # return tf.keras.Model(inputs=[user_id, item_id], outputs=out)

    def init_dense_act_layers(self):
        """
        
        """
        dense_layers = []
        act_layers = []

        layers_dims = self.layers_dims
        for layer_dim in layers_dims:
            dense_layers.append(tf.keras.layers.Dense(units=layer_dim, kernel_regularizer=L2(self.lambda_)))
            act_layers.append(tf.keras.layers.Activation(activation=tf.nn.relu))

        return dense_layers, act_layers



    

class PhilJurisFM:
    def __init__(self, Y, R, num_features=10, epochs=300, epoch_to_rec_at=50, alpha=0.003, lambda_=0.1, regularization="L2"):
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
                # 'train_categorical_accuracy': [],
                # 'val_loss': [],
                # 'val_categorical_accuracy': []
            },
            'epochs': []
        }
        self.cost_per_iter = []

    def train(self, show_vars=True):
        # set utility matrix Y and interaction matrix R to tf constants
        Y, R = self.Y, self.R

        # initialize and get newly initialized params
        X, THETA, BETA = self.init_params()

        # instantiate optimizer
        optimizer = Adam(learning_rate=self.alpha)

        for epoch in range(self.epochs):
            with GradientTape() as tape:
                # multiply resulting dot products to 
                # user-item interaction matrix R
                logits = self.linear(X, THETA, BETA)

                # calculate the cost
                cost = self.J_mean_squared(logits, Y, R) + self.regularizer(X, THETA)

            # take derivative of cost with respect to params X, THETA, and BETA
            grads = tape.gradient(cost, [X, THETA, BETA])

            # apply gradients to params X, THETA, and BETA
            optimizer.apply_gradients(zip(grads, [X, THETA, BETA]))

            

            if ((epoch % self.epoch_to_rec_at) == 0) or (epoch == self.epochs - 1):
                # record all previous values after applying gradients
                self.history['epochs'].append(epoch)
                self.history['history']['train_loss'].append(cost)
                # self.history['history']['train_categorical_accuracy'].append(train_acc)
                # self.history['history']['val_loss'].append(val_cost)
                # self.history['history']['val_categorical_accuracy'].append(val_acc)

                print(f"epoch {epoch} - train_loss: {cost}")
                # print(f"epoch {epoch} - train_loss: {train_cost} - train_categorical_accuracy: {'{:.2%}'.format(train_acc)} - val_loss: {val_cost} - val_categorical_accuracy: {'{:.2%}'.format(val_acc)}")


                if show_vars == True:
                    pass

        # set new coefficients after training
        self.X = X
        self.THETA = THETA
        self.BETA = BETA

        return self.history

    def init_params(self):
        # extract number of users, items, and given arbitrary number of features by user
        n_users, n_items, n_features = self.num_users, self.num_items, self.num_features

        # initialize random values of parameters to optimize from normal distribution
        X = tf.Variable(tf.random.normal(shape=(n_items, n_features), dtype=tf.float64, name='X'))
        THETA = tf.Variable(tf.random.normal(shape=(n_users, n_features), dtype=tf.float64, name='THETA'))
        BETA = tf.Variable(tf.random.normal(shape=(1, n_users), dtype=tf.float64, name='BETA'))

        return X, THETA, BETA

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
            # get the square of all coefficients in each layer excluding biases
            # if 5 layers then loop from 0 to 3 to access all coefficients
            sum_sq_params = tf.reduce_sum(tf.math.square(X)) + tf.reduce_sum(tf.math.square(THETA))

            # multiply by lambda constant then divide by 2
            l2_norm = (self.lambda_ * sum_sq_params) / 2

            # return l2 norm
            return l2_norm

        elif self.regularization.upper() == "L1":
            # if there is only 2 layers then calculation
            # in loop only runs once
            sum_abs_params = tf.reduce_sum(tf.math.abs(X)) + tf.reduce_sum(tf.math.abs(THETA))

            # multiply by lambda constant then divide by 2
            l1_norm = (self.lambda_ * sum_abs_params) / 2

            # return l1 norm
            return l1_norm
        
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


def load_fm_model(n_users, n_items, emb_dim=32, alpha=0.001, lambda_=1):
    """
    this model uses a simple factorization machine architecture where
    embeddings layers are the only layers, and the learned embeddings
    of each user and item is multiplied followed by adding the user 
    embedding biases and item embedding biases to predict the rating
    of the user to that item
    """
    # PHASE 1: model architecture definition
    # since length of user_id is only a scalar. Input would be None, 1 or m x 1
    user_id_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_id')
    item_id_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name='item_id')

    # user and item embedding layer
    user_emb_layer = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_regularizer=L2(lambda_), name='user_embedding')
    item_emb_layer = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_regularizer=L2(lambda_), name='item_embedding')

    # bias vector embedding layer. Note that l2 regularizer 
    # is left our since bias coefficients aren't that all impactful
    user_emb_bias_layer = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros', name='user_embedding_bias')
    item_emb_bias_layer = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer='zeros', name='item_embedding_bias')

    # output layers
    dot_layer = tf.keras.layers.Dot(axes=(2, 1))
    add_layer = tf.keras.layers.Add()

    # PHASE 2: forward pass
    user_emb = user_emb_layer(user_id_input)
    item_emb = item_emb_layer(item_id_input)

    user_emb_bias = user_emb_bias_layer(user_id_input)
    item_emb_bias = item_emb_bias_layer(item_id_input)

    # out = tf.linalg.matmul(user_emb, tf.transpose(item_emb, perm=[0, 2, 1])) + user_emb_bias + item_emb_bias
    user_item_dot = dot_layer([user_emb, tf.transpose(item_emb, perm=[0, 2, 1])])
    out = add_layer([user_item_dot, user_emb_bias, item_emb_bias])

    model = tf.keras.Model(inputs=[user_id_input, item_id_input], outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=alpha),
        loss=mse_loss(),
        metrics=[mse_metric()]
    )

    return model
        

        