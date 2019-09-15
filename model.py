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
		self.model = None
		self.model = self._audio_submodel()
		self.model.summary()
		# self.jointmodel = self._joint_model()
		# self.jointmodel.summary()
        

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
		# model.add(BatchNormalization())
		# model.add(Activation('tanh'))
		# model.add(Dropout(0.4))
		# change number of outputs depending on number of classes
		model.add(Dense(2048, name='dense_2048', trainable=True))
		model.add(BatchNormalization(name='batch_normalization_2048', trainable=True))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		

		# model.add(Dense(256, activation='relu', name='dense_aud2', trainable=False))
		# model.add(Dropout(0.2))
		# model.add(Dense(128, activation='relu', name='dense_aud0', trainable=False))

		# model.add(Dense(100, name='denseX'))
		# model.add(BatchNormalization(name='batch_normalization_2'))
		# model.add(Activation('relu'))


		# jointnet = JointNet()
		# jointmodel = jointnet.model
		# aud_transform = jointnet.audio_submodel
		# for layer in aud_transform.layers:
		# 	layer.trainable = False
		# jointmodel.load_weights('/home/data1/anshulg/proxy_loss_deep3_2048_actual_newdata.keras', by_name=True)


		# model.add(aud_transform)
		# model.add(Dense(500, name='dense_jap1'))
		# model.add(BatchNormalization(name='batch_normalization_jap'))
		# model.add(Activation('relu'))
		model.add(Dense(576, activation='softmax', name='denseY'))
		# model.load_weights('/home/data1/anshulg/models/model_sepspk576_newdata_2048_noisy.h5', by_name=True)


		return model



	def _joint_model(self):


		inp = Input((None, 80))
		out = self.model(inp)

		jointnet = JointNet()
		jointmodel = jointnet.model
		aud_transform = jointnet.audio_submodel
		for layer in aud_transform.layers:
			layer.trainable = False
		jointmodel.load_weights('/home/data1/anshulg/proxy_loss_deep3_2048_actual_newdata.keras', by_name=True)
		aud_transform.add(Dense(576, activation='softmax', name='denseY'))

		finalmodel = Model(inputs=out, outputs=aud_transform.outputs)

		return finalmodel