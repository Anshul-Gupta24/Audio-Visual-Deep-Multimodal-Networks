import pickle
import keras
from keras.models import Model, load_model, Sequential
from keras.layers import InputLayer, Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation
import numpy as np
import os

# import sys
# sys.path.insert(0, '/home/anshulg/WordNet/get_imagenet/functions')
# import imagenet_posterior



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
        # model.add(Dense(576, name='denseY_jap'))

        model.load_weights('/home/data1/anshulg/models/model_sepspk576_newdata_2048_noisy.h5', by_name=True)
        # model.load_weights('/home/data1/anshulg/models/model_sepspk576_jap_2048_subnet_retrain/saved-model-19.hdf5', by_name=True)

        return model



with open('/home/data1/anshulg/google_synth_features_shrink.pkl', 'rb') as fp:
        data = pickle.load(fp) 

# with open('/home/data1/anshulg/google_synth_features_stretch.pkl', 'rb') as fp:
        # data_noisy = pickle.load(fp) 

with open('/home/data1/anshulg/ibm_synth_features_shrink.pkl', 'rb') as fp:
        data_ibm = pickle.load(fp) 

# with open('/home/data1/anshulg/ibm_synth_features_noisy.pkl', 'rb') as fp:
#         data_ibm_noisy = pickle.load(fp) 

with open('/home/data1/anshulg/microsoft_synth_features_shrink.pkl', 'rb') as fp:
        data_ms = pickle.load(fp) 

# with open('/home/data1/anshulg/microsoft_synth_features_noisy.pkl', 'rb') as fp:
        # data_ms_noisy = pickle.load(fp) 


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



# lst_noisy = np.array(list(data_noisy.keys()))
# seq_lens_noisy = {lst_noisy[x]:len(data_noisy[lst_noisy[x]]) for x in range(len(lst_noisy))}
# max_len_noisy = max(list(seq_lens_noisy.values()))			# Get max sequence length
# max_len = max(max_len, max_len_noisy)


# lst_ibm_noisy = np.array(list(data_ibm_noisy.keys()))
# labels_ibm_noisy = [l.split('_')[-2] for l in lst_ibm_noisy]
# seq_lens_ibm_noisy = {lst_ibm_noisy[x]:len(data_ibm_noisy[lst_ibm_noisy[x]]) for x in range(len(lst_ibm_noisy))}
# max_len_ibm_noisy = max(list(seq_lens_ibm_noisy.values()))			# Get max sequence length
# max_len = max(max_len, max_len_ibm_noisy)


# lst_ms_noisy = np.array(list(data_ms_noisy.keys()))
# labels_ms_noisy = [l.split('_')[-2] for l in lst_ms_noisy]
# seq_lens_ms_noisy = {lst_ms_noisy[x]:len(data_ms_noisy[lst_ms_noisy[x]]) for x in range(len(lst_ms_noisy))}
# max_len_ms_noisy = max(list(seq_lens_ms_noisy.values()))			# Get max sequence length
# max_len = max(max_len, max_len_ms_noisy)



# classes = open('/home/anshulg/WordNet/labels_removed2.txt').read().split('\n')
# classes = classes[:-1]
# with open('/home/data1/kiranp/Flickr_selected/feats.pkl', 'rb') as fp:
#         flickr_data = pickle.load(fp)   
classes = open('/home/anshulg/WordNet/labels_exp2.txt','r').read().split('\n')
classes = classes[:-1]


spk_google = range(14)
spk_ibm = range(3)
spk_ms = range(15)
num_speakers = len(spk_google) + len(spk_ibm) + len(spk_ms)
# noise = ['pink_noise', 'white_noise', 'running_tap', 'exercise_bike', 'dude_miaowing', 'doing_the_dishes']

speech_data = {}
for c in classes:
    print(c)
    speech_data[c] = np.zeros((num_speakers, max_len, 80))
    i = 0
    for s in spk_google:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'
        if c=='jack-o-lantern':
            f = "synthesize-audio_jack-o'-lantern_"+str(s)+'.wav'
        
        speech_data[c][i, :seq_lens[f], :] = data[f]
        i += 1

        # for n in noise:
        #         fn = n+'_'+f
        #         speech_data[c][i, :seq_lens_noisy[fn], :] = data_noisy[fn]
        #         i+=1


    for s in spk_ibm:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'
        speech_data[c][i, :seq_lens_ibm[f], :] = data_ibm[f]
        i += 1

#        for n in noise:
#                fn = n+'_'+f
#                speech_data[c][i, :seq_lens_ibm_noisy[fn], :] = data_ibm_noisy[fn]
#                i+=1


    for s in spk_ms:
        f = 'synthesize-audio_'+c+'_'+str(s)+'.wav'
        speech_data[c][i, :seq_lens_ms[f], :] = data_ms[f]
        i += 1

#        for n in noise:
#                fn = n+'_'+f
#                speech_data[c][i, :seq_lens_ms_noisy[fn], :] = data_ms_noisy[fn]
#                i+=1


# with open('/home/data1/anshulg/speech_features_seq_jap_exp.pkl', 'rb') as fp:
#         speech_data = pickle.load(fp) 

print(speech_data['cow'].shape)
input()


with open('/home/data1/anshulg/speech_features_seq_shrink.pkl', 'wb') as fp:
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
