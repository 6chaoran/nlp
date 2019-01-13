import re
import numpy as np
import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

def save_object(obj, filename):
    with open(filename,'wb') as file:
        pickle.dump(obj, file)
        
def load_object(filename):
    with open(filename,'rb') as file:
        res = pickle.load(file)
    return res

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

def tokenize(texts, maxlen = 20, num_words = 9000):
    tokenizer = Tokenizer(filters='',num_words = num_words)
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.index_word) + 1
    max_len = max(list(map(lambda i: len(i.split()), texts)))
    max_len =  min(max_len, maxlen)
    vocab_size = min(vocab_size, num_words)

    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, max_len, padding='post')
    return tokenizer, vocab_size, max_len, padded_seqs