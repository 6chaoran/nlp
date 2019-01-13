import re
import numpy as np
import jieba
from util.preprocess import save_object, load_object

def clean_eng(x):
    x = x.lower()
    x = re.sub('[,.!?]','',x)
    return x

def clean_chn(x):
    x = re.sub('[。，！？\n]','',x)
    x = jieba.cut(x)
    return ' '.join(x)

def load_and_clean_data(filename):
    with open(filename) as file:
        data = []
        for line in file:
            eng, chn = line.split('\t')
            data.append([clean_eng(eng), '<s> '+clean_chn(chn)+' <e>'])
    return np.array(data)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def tokenize(texts):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.index_word) + 1
    max_len = max(list(map(lambda i: len(i.split()), texts)))
    
    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, max_len, padding='post')
    return tokenizer, vocab_size, max_len, padded_seqs

# load data

filename = './data/cmn.txt'

data = load_and_clean_data(filename)
small_data = data[:10000]

src_tokenizer, src_vocab_size, src_max_len, encoder_input_seq = tokenize(small_data[:,0])
tar_tokenizer, tar_vocab_size, tar_max_len, decoder_input_seq = tokenize(small_data[:,1])
decoder_target_seq = decoder_input_seq[:,1:]
decoder_input_seq = decoder_input_seq[:,:-1]
decoder_target_matrix = to_categorical(decoder_target_seq, tar_vocab_size)

save_object({'src_vocab_size': src_vocab_size, 
             'src_max_len': src_max_len, 
             'tar_vocab_size': tar_vocab_size, 
             'tar_max_len': tar_max_len}, 'model.config')

save_object((src_tokenizer, tar_tokenizer), 'tokenizers.pkl')
save_object((encoder_input_seq, decoder_input_seq, decoder_target_matrix), 'inputs.pkl')
