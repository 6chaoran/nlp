import re
import os
import numpy as np
import jieba
import util.util_prep as util
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#===========input parameter ========================
filename = './data/cmn_simplied.txt'
input_path = './input/'
max_len = 15
max_vocab_size = 5000
#===================================================

if not os.path.exists(input_path):
    os.mkdir(input_path)

data = util.load_and_clean_data(filename)
data = data[:10000]

src_tokenizer, src_vocab_size, src_max_len, encoder_input_seq = util.tokenize(data[:,0], max_len, max_vocab_size)
tar_tokenizer, tar_vocab_size, tar_max_len, decoder_input_seq = util.tokenize(data[:,1], max_len, max_vocab_size)
decoder_target_seq = decoder_input_seq[:,1:]
decoder_input_seq = decoder_input_seq[:,:-1]

print('max len: ' + str((src_max_len, tar_max_len)))
print('vocab size: ' + str((src_vocab_size, tar_vocab_size)))

util.save_object({'src_vocab_size': src_vocab_size, 
             'src_max_len': src_max_len, 
             'tar_vocab_size': tar_vocab_size, 
             'tar_max_len': tar_max_len}, input_path+'model_config.pkl')

util.save_object((src_tokenizer, tar_tokenizer), input_path+'tokenizers.pkl')
util.save_object((encoder_input_seq, decoder_input_seq, decoder_target_seq), input_path+'inputs.pkl')
