import tensorflow as tf
from models.Attention.DecoderLayer import DecoderLayer
from models.Attention.attention_utils import positional_encoding

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding_length, rate=0.1):
    super(Decoder, self).__init__() # TODO: see if this does anything

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(10, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding_length, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, look_ahead_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = x + self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask)
      
      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)

    return x, attention_weights