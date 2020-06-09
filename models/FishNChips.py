import tensorflow as tf

from models.ConvBlock import ConvolutionBlock
from models.Attention.Transformer import Transformer

class FishNChips(tf.keras.Model):
    def __init__(self, num_cnn_blocks, max_pool_layer_idx, max_pool_kernel_size, num_layers, d_model, output_dim, num_heads, dff, pe_encoder_max_length, pe_decoder_max_length, rate=0.1):
        super(FishNChips, self).__init__()
        self.pe_encoder_max_length = pe_encoder_max_length
        self.pe_decoder_max_length = pe_decoder_max_length

        # cnn layer for dimensionality expansion
        self.first_cnn = tf.keras.layers.Conv1D(d_model, 1, padding="same", activation="relu", name=f"dimensionality-cnn")
        
        self.max_pool_layer_idx = max_pool_layer_idx
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=max_pool_kernel_size, name="max_pool_1D")
        
        self.cnn_blocks = [ConvolutionBlock([1,3,1], d_model, i) for i in range(num_cnn_blocks)]

        self.transformer = Transformer(num_layers=num_layers, d_model=d_model, output_dim=output_dim, num_heads=num_heads, dff=dff, pe_encoder_max_length=pe_encoder_max_length, pe_decoder_max_length=pe_decoder_max_length)
    
    def call(self, inp, tar, training, look_ahead_mask, use_cached_enc_ouput=False):
        x = self.first_cnn(inp) # to bring to proper dimensionality
        x = self.call_cnn_blocks(x) # won't do anything if no cnn blocks
        att_output, att_weights = self.transformer(x, tar, training, look_ahead_mask, use_cached_enc_ouput)
        return att_output, att_weights

    def call_cnn_blocks(self, x):
        for i,cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
            
            if(i == self.max_pool_layer_idx):
                x = self.max_pool(x)
        return x

    def get_loss(self, real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)