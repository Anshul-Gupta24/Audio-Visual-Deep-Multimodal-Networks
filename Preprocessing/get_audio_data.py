import pickle
import keras
from keras.models import Model, load_model, Sequential
from keras.layers import InputLayer, Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation
import numpy as np
import os


def audio_submodel():

        input_size = 80
        hidden_size = 128

        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))		
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=False))
        model.add(LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=False))
        model.add(Dense(2048, name='dense_2048', trainable=False))
        model.add(BatchNormalization(name='batch_normalization_2048', trainable=False))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.load_weights('../Saved_Models/audio_classifier.h5', by_name=True)
        return model


PATH = '../Data/'
with open(PATH + 'google_synth_features.pkl', 'rb') as fp:
        data = pickle.load(fp) 

with open(PATH + 'ibm_synth_features.pkl', 'rb') as fp:
        data_ibm = pickle.load(fp) 

with open(PATH + 'microsoft_synth_features_shrink.pkl', 'rb') as fp:
        data_ms = pickle.load(fp) 


lst = np.array(list(data.keys()))
seq_lens = {lst[x]:len(data[lst[x]]) for x in range(len(lst))}
max_len = max(list(seq_lens.values()))			# Get max sequence length

lst_ibm = np.array(list(data_ibm.keys()))
labels_ibm = [l.split('_')[-2] for l in lst_ibm]
seq_lens_ibm = {lst_ibm[x]:len(data_ibm[lst_ibm[x]]) for x in range(len(lst_ibm))}
max_len_ibm = max(list(seq_lens_ibm.values()))			# Get max sequence length
max_len = max(max_len, max_len_ibm)

lst_ms = np.array(list(data_ms.keys()))
labels_ms = [l.split('_')[-2] for l in lst_ms]
seq_lens_ms = {lst_ms[x]:len(data_ms[lst_ms[x]]) for x in range(len(lst_ms))}
max_len_ms = max(list(seq_lens_ms.values()))			# Get max sequence length
max_len = max(max_len, max_len_ms)


classes = open('../classes.txt','r').read().split('\n')
classes = classes[:-1]


spk_google = range(14)
spk_ibm = range(3)
spk_ms = range(15)
num_speakers = len(spk_google) + len(spk_ibm) + len(spk_ms)


speech_data = {}
for c in classes:
    print(c)
    speech_data[c] = np.zeros((num_speakers, max_len, 80))
    i = 0
    for s in spk_google:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'        
        speech_data[c][i, :seq_lens[f], :] = data[f]
        i += 1

    for s in spk_ibm:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'
        speech_data[c][i, :seq_lens_ibm[f], :] = data_ibm[f]
        i += 1

    for s in spk_ms:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'
        speech_data[c][i, :seq_lens_ms[f], :] = data_ms[f]
        i += 1


with open('../Data/speech_features_seq.pkl', 'wb') as fp:
        pickle.dump(speech_data, fp) 

# with open('/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_train_3x.pkl', 'rb') as fp:
#         speech_data = pickle.load(fp)


# from keras.applications.xception import Xception
# xception_model = Xception(weights='imagenet')
# model = Sequential()
# model.add(InputLayer((2048, )))
# model.add(Dense(1000))

# model.layers[-1].set_weights(xception_model.layers[-1].get_weights())


# xinds = imagenet_posterior.xception_inds(classes)


# model = audio_submodel()

# speech_features = {}
# num_classes = 90
# num_samples = num_speakers
# for i, c in enumerate(classes):
#         print(i)
#         feats = speech_data[c]
#         feats_100D = model.predict(feats)
#         # feats_576D = np.zeros((num_samples, num_classes))
#         # for i, f in enumerate(feats_100D):
#                 # feats_576D[i] = f[xinds]
#         # print(feats_100D.shape)
#         speech_features[c] = feats_100D

# print(speech_features['gown'].shape)
# input()

# with open('/home/data1/anshulg/speech_features_2048D_eng_exp.pkl', 'wb') as fp:
#         pickle.dump(speech_features, fp) 

