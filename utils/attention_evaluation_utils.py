import tensorflow as tf
import numpy as np

from models.Attention.attention_utils import create_combined_mask
from utils.Other import attentionLabelBaseMap, with_eval_timer

def build(model):
    inp = tf.random.uniform((model.pe_encoder_max_length, 1)) 
    inp = tf.expand_dims(inp, 0) #(1,...,1)
    output = tf.expand_dims([5], 0)
    combined_mask = create_combined_mask(output)
    _, _ = model(inp, output, False, combined_mask)

def evaluate_batch(inp, model, batch_size, as_bases=True):
    
    start_token = 5
    end_token = 6

    output = tf.expand_dims(batch_size*[start_token], 1) # (batchsize, 1) 
    attention_weights = None
    end_tokens = np.zeros(batch_size, dtype=int) # track end token in batch: [0, 1...] -> [no_end_token, end_token, ...] 

    use_cached_enc_output = False
    for _ in range(model.pe_decoder_max_length):
        combined_mask = create_combined_mask(output)
        predictions, attention_weights = model(inp, output, False, combined_mask, use_cached_enc_output) # (batch_size, i + 1, vocab_size)
        use_cached_enc_output = True
        
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size) - take latest prediction 
        predisction_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32) # take highest base token for every batch example
        for j in range(predisction_ids.shape[0]):
            if(predisction_ids[j][0] == end_token):
                end_tokens[j] = 1 # check and add new end tokens
                
        if all(j == 1 for j in end_tokens): # every example in batch has an end token
            output = output[:,1:] # remove start tokens
            output = cut_predition_ends(output, end_token) # cut end token and everything after it
            if as_bases: # convert every example to a string of bases
                output = ["".join([attentionLabelBaseMap[base_token] for base_token in example]) for example in output]
            return output, attention_weights

        output = tf.concat([output, predisction_ids], axis=-1)

    output = output[:,1:]
    output = cut_predition_ends(output, end_token)
    if as_bases:
        output = ["".join([attentionLabelBaseMap[base_token] for base_token in example]) for example in output]
    return output, attention_weights

def cut_predition_ends(output_batch, end_token):
    outputs = []
    for example in output_batch:
        for j,token in enumerate(example):
            if(token == end_token or j == len(example)-1):
                outputs.append(example[:j].numpy())
                break
    return outputs