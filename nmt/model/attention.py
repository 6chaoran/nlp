import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import permute_dimensions, sqrt, constant
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import matrix_band_part

def scaled_dot_product_attention(Q,K,V,mask = False):
    
    assert K.shape.as_list() == V.shape.as_list(), 'shape of K and V must same'
    assert Q.shape.as_list()[2] == K.shape.as_list()[2]
    k = Q.shape.as_list()[2]
    
    ## define layers
    matmul_1 = Dot(axes=-1, name = 'dot_att_matmul1')
    matmul_2 = Dot(axes= 1, name = 'dot_att_matmul2')
    scale = Lambda(lambda x: x / sqrt(constant(k)), name = 'dot_att_scale')
    softmax = Activation(activation='softmax',name = 'dot_att_softmax')
    mask_layer = Lambda(lambda x: matrix_band_part(x, -1 ,0), name = 'dot_att_mask')     # lower tri matrix
    
    y = matmul_1([K,Q])
    y = scale(y)
    y = mask_layer(y)
    y = softmax(y)
    y = matmul_2([y,V])
    return y

def enc_dec_lstm(src_vocab_size, 
    src_max_len, tar_vocab_size, tar_max_len, latent_dim):
    """ encoder-decoder framework using lstm as RNN
    Args:
        src_vocab_size
        src_max_len
        tar_vocab_size 
        tar_max_len
        latent_dim (k)
    Return:
        tuple (train_model, encoder_model, decoder_model)
    """

    # encoder model
    enc_input = Input((None,), name = 'encoder_input_seq')
    enc_embed = Embedding(src_vocab_size + 1, latent_dim, name = 'encoder_embed')
    encoder = LSTM(latent_dim, return_state=True, return_sequences=True, name = 'encoder')

    enc_z, enc_state_h, enc_state_c = encoder(enc_embed(enc_input))
    enc_states = [enc_state_h, enc_state_c]
    enc_model = Model(enc_input, enc_states)

    # decoder model
    dec_input = Input((None,), name = 'decoder_input_seq')
    dec_state_h_input = Input((latent_dim,), name = 'decoder_input_state_h')
    dec_state_c_input = Input((latent_dim,), name = 'decoder_input_state_c')
    dec_states_input = [dec_state_h_input, dec_state_c_input]

    enc_z_input = Input((None, latent_dim))

    dec_embed = Embedding(tar_vocab_size + 1, latent_dim, name = 'decoder_embed')
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, name = 'decoder')
    dec_fc = TimeDistributed(Dense(tar_vocab_size, activation='softmax'), name = 'decoder_output')

    dec_z, dec_state_h, dec_state_c = decoder(dec_embed(dec_input), initial_state = dec_states_input)
    dec_states_output = [dec_state_h, dec_state_c]
    dec_z = scaled_dot_product_attention(dec_z, enc_z_input, enc_z_input, mask = True)
    dec_output = dec_fc(dec_z)

    dec_model = Model([enc_z_input, dec_input]+dec_states_input, [dec_output]+dec_states_output)

    # encoder_decoder training model
    tar_logit, _, _ = decoder(dec_embed(dec_input), initial_state= enc_states)
    tar_logit = scaled_dot_product_attention(tar_logit, enc_z, enc_z)

    tar_output = dec_fc(tar_logit)

    enc_dec_model = Model([enc_input, dec_input], tar_output)
    enc_dec_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return enc_dec_model, enc_model, dec_model

def _init_states(enc_model, src_sentence, tokenizers, src_max_len):
    """generate the states from encoder
    Args:
        enc_model
        src_sentence
        tokenizers: tuple (src_tokenizer, tar_tokenizer)
        src_max_len
    Return:
        tuple (target_triple, initial_states)
    """
    src_tokenizer, tar_tokenizer = tokenizers
    src_index_word = src_tokenizer.index_word
    src_word_index = src_tokenizer.word_index 
    tar_index_word = tar_tokenizer.index_word
    tar_word_index = tar_tokenizer.word_index
    tar_token = '<s>'
    tar_index = tar_word_index.get(tar_token, None)
    if tar_index == None:
        print('start token <s> not found!')
    src_input_seq = src_tokenizer.texts_to_sequences([src_sentence])
    src_input_seq = pad_sequences(src_input_seq, maxlen=src_max_len, padding='post')
    states = enc_model.predict(src_input_seq)
    return ([tar_index], [tar_token], [1.0]), states

def _update_states(dec_model, tar_triple, states, tokenizers):
    """ update the decoder states
    Args:
        dec_model
        tar_triple: (target index[list], target_token[list], target_probability[list])
        states:
        params:
    Return:
        tuple (tar_triple, states)
    """
    src_tokenizer, tar_tokenizer = tokenizers
    src_index_word = src_tokenizer.index_word
    src_word_index = src_tokenizer.word_index 
    tar_index_word = tar_tokenizer.index_word
    tar_word_index = tar_tokenizer.word_index
    tar_index, tar_token, tar_prob = tar_triple
    # predict the token probability, and states
    probs, state_h, state_c = dec_model.predict([[tar_index[-1]]] + states)
    states_new = [state_h, state_c]
    # update the triple
    # greedy search: each time find the most likely token (last position in the sequence)
    probs = probs[0,-1,:]
    tar_index_new = np.argmax(probs)
    tar_token_new = tar_index_word.get(tar_index_new, None)
    tar_prob_new = probs[tar_index_new]
    tar_triple_new = ( 
        tar_index + [tar_index_new],
        tar_token + [tar_token_new],
        tar_prob + [tar_prob_new]
        )
    return tar_triple_new, states_new

def infer_lstm(src_sentence, enc_model, dec_model, tokenizers, max_len = (9,14)):
    """infer the seq2seq model
    Args:
        src_sentence
        enc_model
        dec_model
        tokenizers,
        max_len
    Return:
        decoded sentence
    """
    src_max_len, tar_max_len = max_len
    # initialize with encoder states
    tr, ss = _init_states(enc_model, src_sentence, tokenizers, src_max_len)
    for i in range(tar_max_len):
        # update the triple and states
        tr, ss = _update_states(dec_model, tr, ss, tokenizers)
        if tr[1][-1] == '<e>' or tr[1][-1] == None:
            break
    return ''.join(tr[1])
