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
        Implements the FM (Factorization Machine) architecture by subclassing
        built-in Model
        """
        super(FM, self).__init__()
        # number of unique users and items
        self.n_users = n_users
        self.n_items = n_items

        # hyperparams
        self.lambda_ = lambda_
        self.emb_dim = emb_dim

        # regularization to use
        self.regularization = L2 if regularization == "L2" else L1

        # initialize embedding and embedding bias layers
        self.user_emb_layer = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_regularizer=self.regularization(lambda_), name='user_embedding')
        self.item_emb_layer = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_regularizer=self.regularization(lambda_), name='item_embedding')

        self.user_emb_bias_layer = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros', name='user_embedding_bias')
        self.item_emb_bias_layer = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer='zeros', name='item_embedding_bias')

        # initialize output layer
        self.dot_layer = tf.keras.layers.Dot(axes=(2, 1))
        self.add_layer = tf.keras.layers.Add()


    def call(self, inputs, **kwargs):
        # catch inputs first since Model will be taking in a 2 rows of data
        # the user_id_input which is m x 1 and item_id_input which is m x 1
        # since one example would be one user and one item
        user_id_input = inputs[:, 0]
        item_id_input = inputs[:, 1]

        m = inputs[:, 0].shape[0]

        # DEFINE FORWARD PROPAGATION
        user_emb = self.user_emb_layer(user_id_input)
        item_emb = self.item_emb_layer(item_id_input)

        user_emb_bias = self.user_emb_bias_layer(user_id_input)
        item_emb_bias = self.item_emb_bias_layer(item_id_input)

        # since it is a mere FM model only a linear calculation will be sufficient
        # which is the dot product of the two user_embedding and item_embedding vectors 
        # plus the user_bias and item_bias scalars
        # out = tf.linalg.matmul(user_emb, tf.transpose(item_emb, perm=[0, 2, 1])) + user_emb_bias + item_emb_bias
        user_item_dot = self.dot_layer([tf.expand_dims(user_emb, axis=1), tf.transpose(tf.expand_dims(item_emb, axis=1), perm=[0, 2, 1])])
        out = self.add_layer([tf.reshape(user_item_dot, shape=(m, -1)), user_emb_bias, item_emb_bias])

        return out



class DFM(tf.keras.Model):
    def __init__(self, n_users, n_items, emb_dim=32, layers_dims=[16, 16, 16], lambda_=1, keep_prob=1, regularization="L2"):
        """
        Implements the DFM (Deep Factorization Machine) architecture
        """
        super(DFM, self).__init__()
        # number of unique users and items
        self.n_users = n_users
        self.n_items = n_items

        # hyperparams
        self.lambda_ = lambda_
        self.drop_prob = 1 - keep_prob
        self.emb_dim = emb_dim
        self.layers_dims = layers_dims

        # regularization to use
        self.regularization = L2 if regularization == "L2" else L1

        # number of layers of DNN
        self.num_layers = len(layers_dims)

        # initialize embedding and embedding bias layers
        self.user_emb_layer = tf.keras.layers.Embedding(n_users, emb_dim, embeddings_regularizer=self.regularization(lambda_), name='user_embedding')
        self.item_emb_layer = tf.keras.layers.Embedding(n_items, emb_dim, embeddings_regularizer=self.regularization(lambda_), name='item_embedding')

        self.user_emb_bias_layer = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros', name='user_embedding_bias')
        self.item_emb_bias_layer = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer='zeros', name='item_embedding_bias')

        # initialize dot product layer and add layer for
        # embedding vectors and bias scalars respectively
        self.dot_layer = tf.keras.layers.Dot(axes=(2, 1))
        self.add_layer = tf.keras.layers.Add()

        # initialize flatten layer to flatten sum of the dot product
        # of user_emb & item_emb, user_emb_bias, and  item_emb_bias
        self.flatten_fact_matrix_layer = tf.keras.layers.Flatten()

        # initialize concat layer as input to DNN
        self.concat_layer = tf.keras.layers.Concatenate(axis=2)
        self.flatten_concat_emb_layer = tf.keras.layers.Flatten()

        # initialize dense and activation layers of DNN
        self.dense_layers, self.act_layers, self.dropout_layers = self.init_dense_act_drop_layers()

        # initialize last layer of DNN to dense with no activation
        self.last_dense_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_regularizer=self.regularization(lambda_))

        # output layer will just be an add layer
        self.output_layer = tf.keras.layers.Add()
        

    def call(self, inputs, **kwargs):
        # catch inputs first since Model will be taking in a 2 rows of data
        # the user_id_input which is m x 1 and item_id_input which is m x 1
        # since one example would be one user and one item
        user_id_input = inputs[0]
        item_id_input = inputs[1]
        # print(inputs)

        # DEFINE FORWARD PROPAGATION

        # once user_id_input is passed dimensionality goes from m x 1
        # to m x 1 x emb_dim
        user_emb = self.user_emb_layer(user_id_input)
        item_emb = self.item_emb_layer(item_id_input)

        user_emb_bias = self.user_emb_bias_layer(user_id_input)
        item_emb_bias = self.item_emb_bias_layer(item_id_input)

        # calculate the dot product of the user_emb and item_emb vectors
        user_item_dot = self.dot_layer([user_emb, tf.transpose(item_emb, perm=[0, 2, 1])])
        fact_matrix = self.add_layer([user_item_dot, user_emb_bias, item_emb_bias])
        fact_matrix_flat = self.flatten_fact_matrix_layer(fact_matrix)

        # concatenate the user_emb and item_emb vectors
        # then feed to fully connected deep neural net
        A = self.concat_layer([user_emb, item_emb])
        flat_A = self.flatten_concat_emb_layer(A)

        # forward propagate through deep neural network according to number of layers
        for l in range(self.num_layers):
            # pass concatenated user_embedding and item embedding 
            # to dense layer to calculate Z at layer l
            Z = self.dense_layers[l](flat_A)

            # activate output Z layer by passing to relu activation layer
            flat_A = self.act_layers[l](Z)

            if kwargs['training'] == True:
                flat_A = self.dropout_layers[l](flat_A)

        # pass second to the last layer to a linear layer
        A_last = self.last_dense_layer(flat_A)

        # add the output to the flattened factorized matrix
        out = self.output_layer([A_last, fact_matrix_flat])

        return out

    def init_dense_act_drop_layers(self):
        """
        
        """
        dense_layers = []
        act_layers = []
        dropout_layers = []

        layers_dims = self.layers_dims
        for layer_dim in layers_dims:
            dense_layers.append(tf.keras.layers.Dense(units=layer_dim, kernel_regularizer=self.regularization(self.lambda_)))
            act_layers.append(tf.keras.layers.Activation(activation=tf.nn.relu))

            # drop 1 - keep_prob percent of the neurons e.g. keep_prob
            # is 0.2 so drop 1 - 0.2 or 0.8/80% of the neurons at each 
            # activation layer
            dropout_layers.append(tf.keras.layers.Dropout(rate=self.drop_prob))


        return dense_layers, act_layers, dropout_layers



class MKR(tf.keras.Model):
    def __init__(self, ):
        super(MKR, self).__init__()



        

# acts also as a script for testing
if __name__ == "__main__":
    user_ids = tf.random.uniform(shape=(10, 1), minval=1, maxval=5, dtype=tf.int32)
    item_ids = tf.random.uniform(shape=(10, 1), minval=1, maxval=10, dtype=tf.int32)
    ratings = tf.random.uniform(shape=(10, 1), minval=0.5, maxval=5, dtype=tf.float32)

    # 5 sample unique users and 10 sample unique items
    model = DFM(5, 10)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=mse_loss(),
        metrics=[mse_metric()]
    )

    history = model.fit(
        [user_ids, item_ids],
        ratings,
        epochs=1,
    )
