'''
Creates a dictionary of the form {<class>:feat1, feat2 ... } where feat1, feat2 ... are 2048D audio subnetwork embeddings
'''
import pickle
import keras
from keras.models import Model, load_model, Sequential
from keras.layers import InputLayer, Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def audio_submodel():

        input_size = 80
        hidden_size = 128

        model = Sequential()
        # audio submodel
        model.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))		
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=False))
        model.add(LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=False))
        model.add(Dense(2048, name='dense_2048', trainable=False))
        model.add(BatchNormalization(name='batch_normalization_2048', trainable=False))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.load_weights('../Saved_models/model_audio_subnetwork.h5', by_name=True)
        return model



with open('../Data/speech_features_seq_jap.pkl', 'rb') as fp:
        speech_data = pickle.load(fp) 
classes = open('../new_classes.txt','r').read().split('\n')
classes = classes[:-1]


model = audio_submodel()

speech_features = {}
num_classes = len(classes)
for i, c in enumerate(classes):
        print(i)
        feats = speech_data[c]
        feats_2048D = model.predict(feats)
        speech_features[c] = feats_2048D


with open('../Data/speech_features_2048D_jap.pkl', 'wb') as fp:
        pickle.dump(speech_features, fp) 
