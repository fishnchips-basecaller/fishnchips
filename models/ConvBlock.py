import tensorflow as tf

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, cnn_dims, filters, idx):
        super(ConvolutionBlock, self).__init__()

        self.cnn_layers = [tf.keras.layers.Conv1D(filters, dim, padding="same", activation="relu", use_bias="false", name=f"res{idx}-c{i}") for i,dim in enumerate(cnn_dims)]
        self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in cnn_dims]
        self.activation_layer = tf.keras.layers.Activation('relu', name=f"res{idx}-relu")
        
    def call(self, x):
        res = x
        for cnn_layer, bn_layer in zip(self.cnn_layers, self.bn_layers):
            x = cnn_layer(x)
            x = bn_layer(x)
        
        x += res
        x = self.activation_layer(x)
        return x