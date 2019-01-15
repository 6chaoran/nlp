from model.lstm import enc_dec_lstm, infer_lstm
from util.util_prep import load_object
from util.util_prep import clean_eng
import pandas as pd
import numpy as np
#======== model paramters =======
input_path = './input/'
latent_dim = 512
weight_path = './weight/encoder_decoder_model_weights.h5'
#======== train paramters =======
model_config = load_object(input_path+'model_config.pkl')
tar_vocab_size = model_config['tar_vocab_size']
tokenizers = load_object(input_path+'tokenizers.pkl')

model_config['latent_dim'] = latent_dim
enc_dec_model, enc_model, dec_model = enc_dec_lstm(**model_config)
enc_dec_model.load_weights(weight_path)
tar_max_len = model_config['tar_max_len']


if __name__ == '__main__':

	df = pd.read_csv('./data/cmn_simplied.txt',sep='\t', header=None, names = ['en','cn'])
	enc_dec_model.load_weights(weight_path)

	for i in np.random.choice(len(df), 50, replace=False):
	    src_raw = df.en.values[i]
	    src = clean_eng(src_raw)
	    dec = infer_lstm(src, enc_model, dec_model, tokenizers)
	    print('[%s] => [%s]'%(src,dec))
