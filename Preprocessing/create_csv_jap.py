import numpy as np
import os
import pickle
import pandas as pd


image_train_path = '../Data/img_data_train.pkl'
image_val_path = '../Data/img_data_val.pkl'
image_test_path = '../Data/img_data_test.pkl'
with open(image_train_path, 'rb') as file:
    image_train_feats = pickle.load(file)
with open(image_val_path, 'rb') as file:
    image_val_feats = pickle.load(file)
with open(image_test_path, 'rb') as file:
    image_test_feats = pickle.load(file)
labels = open('../new_classes.txt','r').read().split('\n')
labels = labels[:-1]

print('loaded...')

with open('../Data/spk_ibm_train_jap.pkl', 'rb') as fp:
    spk_ibm_train = pickle.load(fp)

with open('../Data/spk_google_train_jap.pkl', 'rb') as fp:
    spk_google_train = pickle.load(fp)

with open('../Data/spk_microsoft_train_jap.pkl', 'rb') as fp:
    spk_ms_train = pickle.load(fp)

with open('../Data/spk_ibm_val_jap.pkl', 'rb') as fp:
    spk_ibm_val = pickle.load(fp)

with open('../Data/spk_google_val_jap.pkl', 'rb') as fp:
    spk_google_val = pickle.load(fp)

with open('../Data/spk_microsoft_val_jap.pkl', 'rb') as fp:
    spk_ms_val = pickle.load(fp)
    

spk_ibm_train += 1
spk_ms_train += 1 + 
spk_train = np.concatenate((spk_google_train, spk_ibm_train, spk_ms_train))
spk_train = spk_train.astype(int)
spk_ibm_val += 1
spk_ms_val += 1 + 1
spk_val = np.concatenate((spk_google_val, spk_ibm_val, spk_ms_val))
spk_val = spk_val.astype(int)


spk = spk_train
train_set = []
for label in labels:
    for speech_idx in spk:
            train_set.append([label, speech_idx])

df_train = pd.DataFrame(train_set)
print(df_train)
df_train.to_csv('../Data/train_data_proxy_audio_jap.csv')


spk = spk_val
val_set = []
for label in labels:

    # Image anchor
    grounding = 0
    for image_idx in range(len(img_data_val[label])):
            val_set.append([grounding, label, image_idx])
            
    # Audio anchor
    grounding = 1
    for speech_idx in spk:
            # Repeat 5 times to make number divisible by batch size
            for i in range(5):
                val_set.append([grounding, label, speech_idx])


df_val = pd.DataFrame(val_set)
print(df_val)
df_val.to_csv('../Data/val_data_proxy_jap.csv')
