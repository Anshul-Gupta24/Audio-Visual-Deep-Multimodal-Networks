#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from keras.layers import Input, LSTM, Dropout, warnings, Flatten, Dense, TimeDistributed, BatchNormalization, Masking, Activation
from keras.models import Model, Sequential
from keras.engine.topology import get_source_inputs
from keras.utils import get_file, plot_model, layer_utils
from model_proxy2 import JointNet


class WordNet():
	def __init__(self):
		self.model = self._audio_submodel()
		self.model.summary()


	def __call__(self, model_option):
		if model_option == "train":
        		return self.model
		else:
        		return None
    

	def _audio_submodel(self):
        
		hidden_size = 128
		input_size = 80		# change depending on pretrained dnn
		output_classes = 576

		model = Sequential()
		model.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))		
		model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=True))
		model.add(LSTM(hidden_size, activation='tanh', return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=True))
		model.add(Dense(2048, name='dense_2048', trainable=True))
		model.add(BatchNormalization(name='batch_normalization_2048', trainable=True))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(655, activation='softmax', name='denseY'))

		return model
