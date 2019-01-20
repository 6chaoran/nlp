from model.attention import enc_dec_lstm
from util.util_prep import load_object
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#======== model paramters =======
input_path = './input/'
latent_dim = 512
#======== train paramters =======
epochs = 30
batch_size = 128
weight_path = './weight/attention_model_weights.h5'
#================================

model_config = load_object(input_path+'model_config.pkl')
tar_vocab_size = model_config['tar_vocab_size']
tokenizers = load_object(input_path+'tokenizers.pkl')
encoder_input_seq, decoder_input_seq, decoder_target_seq = load_object(input_path+'inputs.pkl')
decoder_target_matrix = to_categorical(decoder_target_seq, tar_vocab_size)

model_config['latent_dim'] = latent_dim
enc_dec_model, enc_mode, dec_model = enc_dec_lstm(**model_config)


if not os.path.exists('./weight/'):
    os.mkdir('./weight/')
    
try:
    enc_dec_model.load_weights(weight_path)
    print('load from previous model')
except:
    print('train a new model')

checkpoint = ModelCheckpoint(filepath=weight_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='val_loss', 
                             verbose = 2)
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=3)
callbacks = [checkpoint, early_stop]

enc_dec_model.fit([encoder_input_seq, decoder_input_seq], decoder_target_matrix,
        batch_size=batch_size,
        epochs=epochs, 
        shuffle = True,
        callbacks=callbacks,
        validation_split=0.1)