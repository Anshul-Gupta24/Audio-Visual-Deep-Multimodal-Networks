from model_proxy2 import JointNet
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
# os.environ['OMP_NUM_THREADS']='8'


if __name__ == '__main__':

    start = timeit.default_timer()
    BATCH_SIZE = 128
    AUD_FEAT = 2048
    IMG_FEAT = 2048
    NUM_CLASSES = 90


    # with open('/home/anshulg/WordNet/proxy_data/spectral_proxy_mat.pkl', 'rb') as fp:
    #     se = pickle.load(fp)


    jointnet = JointNet(np.zeros((576, NUM_CLASSES)))
    model = jointnet.model
    aud_transform = jointnet.audio_submodel
    img_transform = jointnet.image_submodel
    model.load_weights('/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_image/saved-model-24.hdf5', by_name=True)
    # model.load_weights('/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap_image/saved-model-25.hdf5', by_name=True)
    # model.load_weights('/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap2_image/saved-model-40.hdf5', by_name=True)
    print(model.layers[-4].get_weights())
    input()
    # plot_model(model, 'proxymodel.png')


    with open('/home/data1/anshulg/speech_features_2048D_jap_exp.pkl', 'rb') as fp:
        speech_data = pickle.load(fp)

    # max_len = speech_data['abacus'].shape[1]




    with open('/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_train.pkl', 'rb') as fp:
        img_data_train = pickle.load(fp)

    with open('/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_val_3x.pkl', 'rb') as fp:
        img_data_val = pickle.load(fp)


    with open('/home/data1/kiranp/Flickr_selected/feats_train.pkl', 'rb') as fp:
        flickr_train = pickle.load(fp)

    with open('/home/data1/kiranp/Flickr_selected/feats_val.pkl', 'rb') as fp:
        flickr_val = pickle.load(fp)


    # with open('/home/data1/kiranp/extracted_features/imagenet/data_kmeans_val.pkl', 'rb') as fp:
    #     img_data_val = pickle.load(fp)
    classes = open('/home/anshulg/WordNet/labels_removed2.txt','r').read().split('\n')
    classes = classes[:-1]
    label_ind = {c:i for i,c in enumerate(classes)}


    new_classes = open('/home/anshulg/WordNet/labels_exp2.txt','r').read().split('\n')
    new_classes = new_classes[:-1]
    label_ind_new = {c:i for i,c in enumerate(new_classes)}


    df_train = pd.read_csv('/home/anshulg/WordNet/get_imagenet/train_data_proxy_image_exp_new.csv')
    # df_train_image = pd.read_csv('/home/anshulg/WordNet/get_imagenet/train_data_proxy_image_jap.csv')
    # df_val_image = pd.read_csv('/home/anshulg/WordNet/get_imagenet/val_data_proxy_image.csv')
    # df_train_audio = pd.read_csv('/home/anshulg/WordNet/get_imagenet/train_data_proxy_audio_jap.csv')
    # df_val_audio = pd.read_csv('/home/anshulg/WordNet/get_imagenet/val_data_proxy_audio.csv')
    df_val = pd.read_csv('/home/anshulg/WordNet/get_imagenet/val_data_proxy_image_exp_new.csv')

   
    def get_proxy_init(new_classes, transform, indices, data):
        init = np.zeros((576, len(new_classes)))
        for i, c in enumerate(new_classes):
            row = []
            for s in indices:
                row.append(data[c][s])
            pred = transform.predict(np.array(row))           
            # pred_norm = pred / np.expand_dims(np.linalg.norm(pred, axis=1), 1)
            init[:, i] = np.mean(pred, axis=0)

        return init


    init = get_proxy_init(new_classes, img_transform, list(range(30)), flickr_train)
    jointnet = JointNet(init)
    model = jointnet.model
    # model.load_weights('/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_image/saved-model-24.hdf5', by_name=True)
    print(model.layers[-4].get_weights())
    input()


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



    def generator_uni(df, img_data):
        
        num_samples = df.shape[0]
        i = 0
        while True:

            if i%num_samples>=0:
                df = df.sample(frac=1).reset_index(drop=True)                
                i = 0
                
        
            # groundings = df['0'].values[i:i+BATCH_SIZE]
            groundings = np.zeros(BATCH_SIZE)
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



    def generator_unimodal(df_audio, df_image, img_data):
        
        num_samples_audio = df_audio.shape[0]
        num_samples_image = df_image.shape[0]
        # num_batches = (num_samples_audio + num_samples_image) / BATCH_SIZE
        # count = 0
        i_audio = 0
        i_image = 0

        grd = np.random.randint(2)
        while True:
            # count += 1
            # if count == num_batches:
            #     count = 0
            #     grd = np.random.randint(2)
                

            if i_audio%num_samples_audio==0:
                df_audio = df_audio.sample(frac=1).reset_index(drop=True)                
                i_audio = 0

            if i_image%num_samples_image==0:
                df_image = df_image.sample(frac=1).reset_index(drop=True)                
                i_image = 0
                
        
            grd = np.random.randint(2)
            if grd==0:
                df = df_image
                i = i_image
                i_image += BATCH_SIZE
            else:
                df = df_audio
                i = i_audio
                i_audio += BATCH_SIZE
            groundings = np.array([grd] * BATCH_SIZE)

            anchor_labels = df['0'].values[i:i+BATCH_SIZE]
            anchor_indices = df['1'].values[i:i+BATCH_SIZE]
            

            anchor_img = []
            class_mask = np.zeros((BATCH_SIZE, 576))
            class_mask_bar = np.ones((BATCH_SIZE, 576))
            anchor_aud = []
            for s,x in enumerate(groundings):                
                label = anchor_labels[s]
                class_mask[s][label_ind[label]] = 1
                class_mask_bar[s][label_ind[label]] = 0

                # image anchor
                if x==0:                    
                    anchor_img.append(img_data[label][anchor_indices[s]])
                    anchor_aud.append(np.zeros(AUD_FEAT))
                   
                    
                # audio anchor
                else:                   
                    anchor_aud.append(speech_data[label][anchor_indices[s]])
                    anchor_img.append(np.zeros(576))

                               
            yield [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(BATCH_SIZE)
            
            i += BATCH_SIZE




    folder = "/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap2_image"
    if not os.path.isdir(folder):
        os.system('mkdir '+folder)
    filepath = folder+"/saved-model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, period=1)
    callbacks_list = [checkpoint]


    # NOTE: val set consists of seen images and unseen audio so we are repeating img_data_train for val generator

    num_train_batches = ((df_train.shape[0]) / BATCH_SIZE)
    num_val_batches = ((df_val.shape[0]) / BATCH_SIZE)
    history = model.fit_generator(generator_uni(df_train, flickr_train), steps_per_epoch=num_train_batches, epochs=100, callbacks=callbacks_list, validation_data=generator_uni(df_val, flickr_val), initial_epoch=0, validation_steps=num_val_batches)


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("proxy2_loss2_deep_softmax_relu4_image.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

   
    with open('/home/data1/anshulg/history_proxy2_loss2_deep_softmax_relu4_image.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    time = open('time.txt','w')
    time.write(str(stop-start))
    