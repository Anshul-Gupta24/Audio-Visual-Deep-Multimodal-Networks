'''
Creates a dictionary of the form {<class>:feat1, feat2 ... } where feat1, feat2 ... are the LRE baseline features
'''
import pickle
import numpy as np
import os


PATH = '../Data/'
with open(PATH + 'google_synth_features_jap.pkl', 'rb') as fp:
        data = pickle.load(fp) 

with open(PATH + 'ibm_synth_features_jap.pkl', 'rb') as fp:
        data_ibm = pickle.load(fp) 

with open(PATH + 'microsoft_synth_features_jap.pkl', 'rb') as fp:
        data_ms = pickle.load(fp) 


lst = np.array(list(data.keys()))
seq_lens = {lst[x]:len(data[lst[x]]) for x in range(len(lst))}
max_len = max(list(seq_lens.values()))			# Get max sequence length

lst_ibm = np.array(list(data_ibm.keys()))
seq_lens_ibm = {lst_ibm[x]:len(data_ibm[lst_ibm[x]]) for x in range(len(lst_ibm))}
max_len_ibm = max(list(seq_lens_ibm.values()))			# Get max sequence length
max_len = max(max_len, max_len_ibm)

lst_ms = np.array(list(data_ms.keys()))
seq_lens_ms = {lst_ms[x]:len(data_ms[lst_ms[x]]) for x in range(len(lst_ms))}
max_len_ms = max(list(seq_lens_ms.values()))			# Get max sequence length
max_len = max(max_len, max_len_ms)


classes = open('../new_classes.txt','r').read().split('\n')
classes = classes[:-1]


spk_google = range(1)
spk_ibm = range(1)
spk_ms = range(3)
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


with open(PATH + 'speech_features_seq_jap.pkl', 'wb') as fp:
        pickle.dump(speech_data, fp) 
