import tensorflow as tf
from tensorflow.keras.regularizers import L1, L2



class CrossAndCompress(tf.keras.layers.Layer):
    def __init__(self, weight_regularizer, **kwargs):
        """
        args:
            weight_regularizer - can be is a tf.keras.regularizers.L1 or
            tf.keras.regularizers.L2
        """
        super(CrossAndCompress, self).__init__()
        # passing emb_dim doesn't seem right since call would happen like this
        # CrossAndCompress(emb_dim=emb_dim_of_model_class)
        # hmm or maybe not, but maybe yes since CrossAndCompress doesn't really ask like Dense
        # does for the number of units or in this case emb_dim. Let's jsut figure it out at run time

        # gets an instance of the passed regularizer object from tf
        # can be L2(1) and L1(1) or 'L2' and 'L1'
        self.weight_regularizer = tf.keras.regularizers.get(weight_regularizer)



    def build(self, input_shape):
        """
        args:
            input_shape - the default argument of the method self.build() 
            from the Layer class of tf-keras, where it contains the shape
            of the batch passed through the self.call() of the Layer class
            when it is invoked

        """
        # sample the input at index 0 since we have two inputs to
        # this layer then get the number of dimensions of an input
        # examples feature vector representation, which is at -1th
        # index or the last index position
        print(input_shape)
        emb_dim = input_shape[0][-1]
        print(emb_dim)

        self.theta_vv = self.add_weight(
            shape=(1, emb_dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=self.weight_regularizer,
            trainable=True,
            name='theta_vv')
        
        self.theta_ev = self.add_weight(
            shape=(1, emb_dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=self.weight_regularizer,
            trainable=True,
            name='theta_ev')
        
        self.theta_ve = self.add_weight(
            shape=(1, emb_dim, 1), 
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=self.weight_regularizer,
            trainable=True,
            name='theta_ve')
        
        self.theta_ee = self.add_weight(
            shape=(1, emb_dim, 1), 
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=self.weight_regularizer,
            trainable=True,
            name='theta_ee')
        
        self.beta_v = self.add_weight(
            shape=(1, emb_dim, 1), 
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name='beta_v')
        
        self.beta_e = self.add_weight(
            shape=(1, emb_dim, 1),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name='beta_e')
    
    def call(self, inputs, *args, **kwargs):
        """
        
        """
        # print(inputs)
        enc_user, enc_item = inputs
        # print(enc_user, enc_item)

if __name__ == "__main__":
    enc_users = tf.ones(shape=(10, 1, 64), dtype=tf.int32)
    enc_items = tf.ones(shape=(10, 1, 64), dtype=tf.int32)

    # when call is invoked it is passed the whole dataset or
    # the whole batch during training which will have shape 
    # [m x 1 x 64, m x 1 x 64] ( or [(None, 1, 64), (None, 1, 64)])
    # why it is enclosed in square brackets is because our inputs
    # are also enclosed in square brackets e.g. [enc_users, enc_items]
    # and when inputs are represented as their respective shapes it returns
    # [(None, 1, 64), (None, 1, 64)]
    cc_layer = CrossAndCompress(weight_regularizer=L2(1))
    cc_layer(inputs=[enc_users, enc_items])

