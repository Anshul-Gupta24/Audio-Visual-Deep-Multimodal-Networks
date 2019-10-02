'''
Code to train the proxy based deep multimodal network for Japanese speech-image input
'''
from model_proxy import JointNet
import keras	
import numpy as np
from keras.utils import plot_model
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import timeit
import json
import pandas as pd
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


if __name__ == '__main__':

    start = timeit.default_timer()
    BATCH_SIZE = 5
    AUD_FEAT = 2048
    IMG_FEAT = 2048
    NUM_CLASSES = 90
    num_samples_train = 15
    num_samples_val = 55


    jointnet = JointNet(np.zeros((576, NUM_CLASSES)))
    model = jointnet.model
    aud_transform = jointnet.audio_submodel
    model.load_weights('Saved_models/model_proxy/saved-model-24.hdf5', by_name=True)    # Choose epoch weights to load


    with open('Data/speech_features_2048D_jap.pkl', 'rb') as fp:
        speech_data = pickle.load(fp)

    with open('Data/image_features_train.pkl', 'rb') as fp:
        img_data_train = pickle.load(fp)

    with open('Data/image_features_val.pkl', 'rb') as fp:
        img_data_val = pickle.load(fp)


    new_classes = open('new_classes.txt','r').read().split('\n')
    new_classes = new_classes[:-1]
    label_ind_new = {c:i for i,c in enumerate(new_classes)}


    df_train = pd.read_csv('Data/train_data_proxy_audio_jap.csv')
    df_val = pd.read_csv('Data/val_data_proxy_jap.csv')
    
    
    def generator(df, img_data, i, num_samples):
        
        groundings = df['0'].values[i:i+num_samples]
        anchor_labels = df['1'].values[i:i+num_samples]
        anchor_indices = df['2'].values[i:i+num_samples]
        

        anchor_img = []
        class_mask = np.zeros((num_samples, NUM_CLASSES))
        class_mask_bar = np.ones((num_samples, NUM_CLASSES))
        anchor_aud = []
        for s,x in enumerate(groundings):                
            label = anchor_labels[s]
            class_mask[s][label_ind_new[label]] = 1
            class_mask_bar[s][label_ind_new[label]] = 0

            # image anchor
            if x==0:                    
                anchor_img.append(img_data[label][anchor_indices[s]])
                anchor_aud.append(np.zeros(AUD_FEAT))
                # anchor_aud.append(np.zeros((max_len, AUD_FEAT)))                
                
            # audio anchor
            else:                   
                anchor_aud.append(speech_data[label][anchor_indices[s]])
                anchor_img.append(np.zeros(IMG_FEAT))

                               
        return [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(num_samples)



    def generator_uni(df, img_data, i, num_samples):
        
        groundings = np.ones(num_samples)
        anchor_labels = df['0'].values[i:i+num_samples]
        anchor_indices = df['1'].values[i:i+num_samples]
        

        anchor_img = []
        class_mask = np.zeros((num_samples, NUM_CLASSES))
        class_mask_bar = np.ones((num_samples, NUM_CLASSES))
        anchor_aud = []
        for s,x in enumerate(groundings):                
            label = anchor_labels[s]
            class_mask[s][label_ind_new[label]] = 1
            class_mask_bar[s][label_ind_new[label]] = 0

            # image anchor
            if x==0:                    
                anchor_img.append(img_data[label][anchor_indices[s]])
                anchor_aud.append(np.zeros(AUD_FEAT))
                
                
            # audio anchor
            else:                   
                anchor_aud.append(speech_data[label][anchor_indices[s]])
                anchor_img.append(np.zeros(IMG_FEAT))

                               
        return [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(num_samples)


    i = 0
    j = 0
    folder = "Saved_models/model_proxy_jap"
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
    for x in range(18):

        model.load_weights('Saved_models/model_proxy/saved-model-24.hdf5', by_name=True)    # For fresh start
        model.compile(loss=jointnet.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-5))

        folder = "Saved_models/model_proxy_jap/block_"+str(x)
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
        filepath = folder+"/saved-model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min', period=1)
        callbacks_list = [checkpoint]


        num_train_batches = ((df_train.shape[0]) / BATCH_SIZE)
        num_val_batches = ((df_val.shape[0]) / BATCH_SIZE)

        X, Y = generator_uni(df_train, flickr_train, i, num_samples_train)
        X_val, Y_val = generator(df_val, flickr_val, i, num_samples_val)

        history = model.fit(X, Y, batch_size=BATCH_SIZE, epochs=20, callbacks=callbacks_list, validation_data=(X_val, Y_val), shuffle=True)

        i += num_samples_train
        j += num_samples_val
    
