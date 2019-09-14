'''
Code to train the audio subnetwork
'''

from model import WordNet
import kiran_model
import keras	
import numpy as np
from keras.utils import plot_model
from keras.models import load_model
import os
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import timeit
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

FEATURE_SIZE = 80 
BATCH_SIZE = 32


def get_train_test_ind(lst, spk_train, spk_val, spk_test=[]):
    
    train_ind = []
    val_ind = []
    test_ind = []

    for i in range(len(lst)):
        spk = int(lst[i].split('_')[-1][:-4])

        if(spk in spk_train):
            train_ind.append(i)
        elif(spk in spk_val):
            val_ind.append(i)
        elif(spk in spk_test):
            test_ind.append(i)

    return np.array(train_ind), np.array(val_ind), np.array(test_ind)


def load_data(path, classes, class_enc):

    with open(path, 'rb') as fp:
        data = pickle.load(fp) 

    lst = np.array(list(data.keys()))
    labels = np.array([l.split('_')[-2] for l in lst])
    labels_ind_valid = [i for i,l in enumerate(labels) if l in classes]
    lst = lst[labels_ind_valid]
    labels = labels[labels_ind_valid]
  
    label_vecs = np.zeros((len(lst), len(classes)))
    for i,lv in enumerate(label_vecs):
        lv[class_enc[labels[i]]] = 1

    seq_lens = [len(data[lst[x]]) for x in range(len(lst))]
    max_len = max(seq_lens)			# Get max sequence length


    return data, lst, label_vecs, seq_lens, max_len


def get_data():

    classes = open('classes.txt','r').read().split('\n')
    classes = classes[:-1]
    print(len(classes))
    class_enc = {c:i for i,c in enumerate(classes)}


    # Google
    data, lst, label_vecs, seq_lens, max_len = load_data('Data/google_synth_features', classes, class_enc)
    # IBM
    data_ibm, lst_ibm, label_vecs_ibm, seq_lens_ibm, max_len_ibm = load_data('Data/ibm_synth_features.pkl', classes, class_enc)
    # Microsoft
    data_ms, lst_ms, label_vecs_ms, seq_lens_ms, max_len_ms = load_data('Data/microsoft_synth_features.pkl', classes, class_enc)
       
    max_len = np.max([max_len, max_len_ibm, max_len_ms])
  

    # Get training and validation data

    data_arr = np.zeros((len(lst), max_len, FEATURE_SIZE))
    for x in range(len(lst)):
        data_arr[x, :seq_lens[x], :] = data[lst[x]]    

    data_arr_ibm = np.zeros((len(lst_ibm), max_len, FEATURE_SIZE))
    for x in range(len(lst_ibm)):
        data_arr_ibm[x, :seq_lens_ibm[x], :] = data_ibm[lst_ibm[x]]     

    data_arr_ms = np.zeros((len(lst_ms), max_len, FEATURE_SIZE))
    for x in range(len(lst_ms)):
        data_arr_ms[x, :seq_lens_ms[x], :] = data_ms[lst_ms[x]]    

    data_arr = np.concatenate((data_arr, data_arr_ibm, data_arr_ms), axis=0)
    label_vecs = np.concatenate((label_vecs, label_vecs_ibm, label_vecs_ms))
    seq_lens = np.concatenate((seq_lens, seq_lens_ibm, seq_lens_ms))

    
    with open('Data/spk_ibm_train.pkl', 'rb') as fp:
        spk_ibm_train = pickle.load(fp)

    with open('Data/spk_google_train.pkl', 'rb') as fp:
        spk_google_train = pickle.load(fp)

    with open('Data/spk_microsoft_train.pkl', 'rb') as fp:
        spk_ms_train = pickle.load(fp)

    with open('Data/spk_ibm_val.pkl', 'rb') as fp:
        spk_ibm_val = pickle.load(fp)

    with open('Data/spk_google_val.pkl', 'rb') as fp:
        spk_google_val = pickle.load(fp)

    with open('Data/spk_microsoft_val.pkl', 'rb') as fp:
        spk_ms_val = pickle.load(fp)

        
    train_ind, val_ind, _ = get_train_test_ind(lst, spk_google_train, spk_google_val)
    train_ind_ibm, val_ind_ibm, _ = get_train_test_ind(lst_ibm, spk_ibm_train, spk_ibm_val)
    train_ind_ms, val_ind_ms, _ = get_train_test_ind(lst_ms, spk_ms_train, spk_ms_val)


    train_ind_ibm += (len(lst))
    train_ind_ms += (len(lst) len(lst_ibm))
    val_ind_ibm += (len(lst))
    val_ind_ms += (len(lst) + len(lst_ibm))

    train_ind = np.concatenate((train_ind, train_ind_ibm, train_ind_ms))
    val_ind = np.concatenate((val_ind, val_ind_ibm, val_ind_ms))
    val_ind = val_ind.astype(int)


    data_arr_train = data_arr[train_ind]
    data_arr_val = data_arr[val_ind]

    label_vecs_train = label_vecs[train_ind]
    label_vecs_val = label_vecs[val_ind]

    seq_lens_train = seq_lens[train_ind]
    seq_lens_val = seq_lens[val_ind]


    return data_arr_train, data_arr_val, label_vecs_train, label_vecs_val
    


if __name__ == '__main__':

    model = WordNet()("train")
    model.compile(optimizer=keras.optimizers.Adam(lr=5e-4, decay=5e-6), 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    data_arr_train, data_arr_val, label_vecs_train, label_vecs_val = get_data()

   
    def generator(data_arr, label_vecs):
        batch_size = BATCH_SIZE
        features = FEATURE_SIZE
        i = 0
        num_data = data_arr.shape[0]
        indices = np.array(range(num_data))
        np.random.shuffle(indices)
        while(True):

            j=i+batch_size
            if j%len(data_arr)>0:
                i = 0
                j=i+batch_size
                # shuffle data
                np.random.shuffle(indices)

            batch_data = data_arr[indices[i:j]]
            batch_labels = label_vecs[indices[i:j]]

            i=j
                        
            yield batch_data, batch_labels
        

    filepath = "Saved_models/model_audio_subnetwork.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]


    train_size = len(data_arr_train)
    val_size = len(data_arr_val)


    history = model.fit_generator(generator(data_arr_train, label_vecs_train), steps_per_epoch=int(train_size/BATCH_SIZE), epochs=40, callbacks=callbacks_list, validation_data=generator(data_arr_val, label_vecs_val), validation_steps=int(val_size/BATCH_SIZE))


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("Plots/model_loss_audio_subnetwork.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig("Plots/model_acc_audio_subnetwork.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()
