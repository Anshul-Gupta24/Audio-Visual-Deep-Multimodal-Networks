'''
Code to train the proxy based deep multimodal network
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
    BATCH_SIZE = 128
    AUD_FEAT = 2048
    IMG_FEAT = 2048
    NUM_CLASSES = 90


    jointnet = JointNet(np.zeros((576, NUM_CLASSES)))
    model = jointnet.model


    with open('Data/speech_features_2048D.pkl', 'rb') as fp:
        speech_data = pickle.load(fp)

    with open('Data/image_features_train.pkl', 'rb') as fp:
        img_data_train = pickle.load(fp)

    with open('Data/image_features_val.pkl', 'rb') as fp:
        img_data_val = pickle.load(fp)


    classes = open('classes.txt','r').read().split('\n')
    classes = classes[:-1]
    label_ind = {c:i for i,c in enumerate(classes)}


    df_train_aud = pd.read_csv('Data/train_data_proxy_audio.csv')
    df_train_img = pd.read_csv('Data/train_data_proxy_image.csv')
    df_val = pd.read_csv('Data/val_data_proxy.csv')

    
    def generator_uni(df, img_data, grd):
        
        num_samples = df.shape[0]
        i = 0
        while True:

            if i%num_samples>=0:
                df = df.sample(frac=1).reset_index(drop=True)                
                i = 0
                
        
            if grd==0:
                groundings = np.zeros(BATCH_SIZE)
            else:
                groundings = np.ones(BATCH_SIZE)
            anchor_labels = df['0'].values[i:i+BATCH_SIZE]
            anchor_indices = df['1'].values[i:i+BATCH_SIZE]
            

            anchor_img = []
            class_mask = np.zeros((BATCH_SIZE, NUM_CLASSES))
            class_mask_bar = np.ones((BATCH_SIZE, NUM_CLASSES))
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

                               
            yield [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(BATCH_SIZE)
            
            i += BATCH_SIZE
    

    def generator(df, img_data):
        
        num_samples = df.shape[0]
        i = 0
        while True:

            if i%num_samples>=0:
                df = df.sample(frac=1).reset_index(drop=True)                
                i = 0
                

            groundings = df['0'].values[i:i+BATCH_SIZE]
            anchor_labels = df['1'].values[i:i+BATCH_SIZE]
            anchor_indices = df['2'].values[i:i+BATCH_SIZE]
            

            anchor_img = []
            class_mask = np.zeros((BATCH_SIZE, NUM_CLASSES))
            class_mask_bar = np.ones((BATCH_SIZE, NUM_CLASSES))
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

                               
            yield [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(BATCH_SIZE)
            
            i += BATCH_SIZE


   
    num_train_batches = ((df_train.shape[0]) / BATCH_SIZE)
    num_val_batches = ((df_val.shape[0]) / BATCH_SIZE)
    
    # Phase1: Train only on images
    history = model.fit_generator(generator_uni(df_train, img_data_train, 0), steps_per_epoch=num_train_batches, epochs=20, validation_data=generator(df_val, img_data_val), initial_epoch=0, validation_steps=num_val_batches)

    folder = "Saved_models/model_proxy"
    if not os.path.isdir(folder):
        os.system('mkdir '+folder)
    filepath = folder+"/saved-model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, period=1)
    callbacks_list = [checkpoint]
    
    # Phase2: Train only on audio
    history = model.fit_generator(generator_uni(df_train, img_data_train, 1), steps_per_epoch=num_train_batches, epochs=100, callbacks=callbacks_list, validation_data=generator(df_val, img_data_val), initial_epoch=0, validation_steps=num_val_batches)


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("Plots/model_proxy_loss.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    
