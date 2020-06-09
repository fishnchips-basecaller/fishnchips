import tensorflow as tf
from models.Attention.attention_utils import positional_encoding
from models.Attention.EncoderLayer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding_length, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding_length, self.d_model)
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training):
    seq_len = tf.shape(x)[1]
    
    # adding embedding
    # we removed this since we think we don't need it
    # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    # TODO:
    # We think this has to do with the embedding small number output
    # Chceck this when it works to see if the numbers still look ok
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    # add possitional encoding
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)
    
    return x  # (batch_size, input_seq_len, d_model)