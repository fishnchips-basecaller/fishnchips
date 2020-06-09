import tensorflow as tf
from models.Attention.Encoder import Encoder
from models.Attention.Decoder import Decoder

class Transformer(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, output_dim, num_heads, dff, pe_encoder_max_length, pe_decoder_max_length, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_encoder_max_length, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_decoder_max_length, rate)
    self.final_layer = tf.keras.layers.Dense(output_dim)

    self.prev_inp = None
    self.cached_enc_output = None
    
  def call(self, inp, tar, training, look_ahead_mask, use_cached_enc_ouput=False):         
    enc_output = self._call_encoder(inp, training, use_cached_enc_ouput)
    dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask) # (batch_size, tar_seq_len, d_model), weights
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    return final_output, attention_weights

  def _call_encoder(self, inp, training, use_cached_enc_ouput):
    if training: # if we are training we alway run the encoder
      return self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

    if use_cached_enc_ouput:
      return self.cached_enc_output
        
    enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)
    self.cached_enc_output = enc_output
    self.prev_inp = inp
    return enc_output