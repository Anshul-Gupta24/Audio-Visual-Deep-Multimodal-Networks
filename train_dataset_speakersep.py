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


def get_mean_std(data_arr, seq_lens):

    total_len = np.sum(seq_lens)

    unrolled = np.zeros((total_len, FEATURE_SIZE))
    start = 0
    end = 0
    for i in range(len(data_arr)):
        end += seq_lens[i]
        unrolled[start:end,:] = np.split(data_arr[i], data_arr.shape[1])[:seq_lens[i]]        
        start = end

    mean = np.mean(unrolled, axis=0)
    std = np.std(unrolled, axis=0)

    return mean, std


def get_train_test_spk(total, train, val, test):

    spk_rem = range(total)
    spk_train = np.random.choice(spk_rem, train, replace=False)
    spk_rem = list(set(spk_rem) - set(spk_train))
    spk_val = np.random.choice(spk_rem, val, replace=False)
    spk_rem = list(set(spk_rem) - set(spk_val))
    spk_test = np.random.choice(spk_rem, test, replace=False)

    return spk_train, spk_val, spk_test


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

    classes = open('/home/anshulg/WordNet/labels_removed2.txt','r').read().split('\n')
    classes = classes[:-1]
    # df = pd.read_csv('/home/anshulg/WordNet/get_imagenet/functions/imagenet-katakana.csv')
    # katas = df['kata'].values
    # katakana_eng = df['eng'].values[katas=='katakana']
    # katakana = df['jap'].values[katas=='katakana']
    # classes = katakana_eng
    print(len(classes))

    class_enc = {c:i for i,c in enumerate(classes)}
    # with open('/home/anshulg/WordNet/get_imagenet/functions/label_to_ind576.pkl','wb') as fp:
    #     pickle.dump(class_enc, fp)
    # class_enc["jack-o'-lantern"] = class_enc["jack-o-lantern"]      # discrepency in Google and IBM
    

    # Google
    data, lst, label_vecs, seq_lens, max_len = load_data('/home/data1/anshulg/google_synth_features', classes, class_enc)
    # Google noisy
    data_noisy, lst_noisy, label_vecs_noisy, seq_lens_noisy, max_len_noisy = load_data('/home/data1/anshulg/google_synth_features_noisy', classes, class_enc)
    # IBM
    data_ibm, lst_ibm, label_vecs_ibm, seq_lens_ibm, max_len_ibm = load_data('/home/data1/anshulg/ibm_synth_features.pkl', classes, class_enc)
    # IBM noisy
    data_ibm_noisy, lst_ibm_noisy, label_vecs_ibm_noisy, seq_lens_ibm_noisy, max_len_ibm_noisy = load_data('/home/data1/anshulg/ibm_synth_features_noisy.pkl', classes, class_enc)
    # Microsoft
    data_ms, lst_ms, label_vecs_ms, seq_lens_ms, max_len_ms = load_data('/home/data1/anshulg/microsoft_synth_features.pkl', classes, class_enc)
    # Microsoft noisy
    data_ms_noisy, lst_ms_noisy, label_vecs_ms_noisy, seq_lens_ms_noisy, max_len_ms_noisy = load_data('/home/data1/anshulg/microsoft_synth_features_noisy.pkl', classes, class_enc)
    
    max_len = np.max([max_len, max_len_noisy, max_len_ibm, max_len_ibm_noisy, max_len_ms, max_len_ms_noisy])

   
    
    # Real Data

    # with open('/home/data1/anshulg/google_synth_features2', 'rb') as fp:
    #     data_real = pickle.load(fp) 

    # lst_real = np.array(list(data_real.keys()))
    # labels_real = np.array([l.split('_')[-2] for l in lst_real])
    # number_real = np.array([int(l.split('_')[-1][:-4]) for l in lst_real])
    # labels_ind_valid = [i for i,l in enumerate(number_real) if l>13]
    # lst_real = lst_real[labels_ind_valid]
    # labels_real = labels_real[labels_ind_valid]
    # labels_ind_valid = [i for i,l in enumerate(labels_real) if l in classes]
    # lst_real = lst_real[labels_ind_valid]
    # labels_real = labels_real[labels_ind_valid]

  
    # label_vecs_real = np.zeros((len(lst_real), len(classes)))
    # for i,lv in enumerate(label_vecs_real):
    #     lv[class_enc[labels_real[i]]] = 1

    # seq_lens_real = [len(data_real[lst_real[x]]) for x in range(len(lst_real))]
    # max_len_real = max(seq_lens_real)			# Get max sequence length
    # max_len = max(max_len, max_len_real)



    # Get training and validation data
    # Keep some speakers only in validation set

    data_arr = np.zeros((len(lst), max_len, FEATURE_SIZE))
    for x in range(len(lst)):
        data_arr[x, :seq_lens[x], :] = data[lst[x]]    

    data_arr_noisy = np.zeros((len(lst_noisy), max_len, FEATURE_SIZE))
    for x in range(len(lst_noisy)):
        data_arr_noisy[x, :seq_lens_noisy[x], :] = data_noisy[lst_noisy[x]]    

    data_arr_ibm = np.zeros((len(lst_ibm), max_len, FEATURE_SIZE))
    for x in range(len(lst_ibm)):
        data_arr_ibm[x, :seq_lens_ibm[x], :] = data_ibm[lst_ibm[x]]    

    data_arr_ibm_noisy = np.zeros((len(lst_ibm_noisy), max_len, FEATURE_SIZE))
    for x in range(len(lst_ibm_noisy)):
        data_arr_ibm_noisy[x, :seq_lens_ibm_noisy[x], :] = data_ibm_noisy[lst_ibm_noisy[x]]    

    data_arr_ms = np.zeros((len(lst_ms), max_len, FEATURE_SIZE))
    for x in range(len(lst_ms)):
        data_arr_ms[x, :seq_lens_ms[x], :] = data_ms[lst_ms[x]]    

    data_arr_ms_noisy = np.zeros((len(lst_ms_noisy), max_len, FEATURE_SIZE))
    for x in range(len(lst_ms_noisy)):
        data_arr_ms_noisy[x, :seq_lens_ms_noisy[x], :] = data_ms_noisy[lst_ms_noisy[x]]  


    # data_arr_real = np.zeros((len(lst_real), max_len, FEATURE_SIZE))
    # for x in range(len(lst_real)):
    #     data_arr_real[x, :seq_lens_real[x], :] = data_real[lst_real[x]] 



    data_arr = np.concatenate((data_arr, data_arr_noisy, data_arr_ibm, data_arr_ibm_noisy, data_arr_ms, data_arr_ms_noisy), axis=0)
    label_vecs = np.concatenate((label_vecs, label_vecs_noisy, label_vecs_ibm, label_vecs_ibm_noisy, label_vecs_ms, label_vecs_ms_noisy))
    seq_lens = np.concatenate((seq_lens, seq_lens_noisy, seq_lens_ibm, seq_lens_ibm_noisy, seq_lens_ms, seq_lens_ms_noisy))

    
    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_train.pkl', 'rb') as fp:
        spk_ibm_train = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_train.pkl', 'rb') as fp:
        spk_google_train = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_train.pkl', 'rb') as fp:
        spk_ms_train = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_val.pkl', 'rb') as fp:
        spk_ibm_val = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_val.pkl', 'rb') as fp:
        spk_google_val = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_val.pkl', 'rb') as fp:
        spk_ms_val = pickle.load(fp)


    # spk_google_train = [0]
    # spk_google_val = []
    # spk_ibm_train = [0]
    # spk_ibm_val = []
    # spk_ms_train = [1]
    # spk_ms_val = [0]

    train_ind, val_ind, _ = get_train_test_ind(lst, spk_google_train, spk_google_val)
    train_ind_noisy, val_ind_noisy, _ = get_train_test_ind(lst_noisy, spk_google_train, spk_google_val)
    train_ind_ibm, val_ind_ibm, _ = get_train_test_ind(lst_ibm, spk_ibm_train, spk_ibm_val)
    train_ind_ibm_noisy, val_ind_ibm_noisy, _ = get_train_test_ind(lst_ibm_noisy, spk_ibm_train, spk_ibm_val)
    train_ind_ms, val_ind_ms, _ = get_train_test_ind(lst_ms, spk_ms_train, spk_ms_val)
    train_ind_ms_noisy, val_ind_ms_noisy, _ = get_train_test_ind(lst_ms_noisy, spk_ms_train, spk_ms_val)
    # train_ind_real = np.arange(len(lst_real))

    train_ind_noisy += len(lst)
    train_ind_ibm += (len(lst) + len(lst_noisy))
    train_ind_ibm_noisy += (len(lst) + len(lst_noisy) + len(lst_ibm))
    train_ind_ms += (len(lst) + len(lst_noisy) + len(lst_ibm) + len(lst_ibm_noisy))
    train_ind_ms_noisy += (len(lst) + len(lst_noisy) + len(lst_ibm) + len(lst_ibm_noisy) + len(lst_ms))
    # train_ind_real += (len(lst) + len(lst_noisy) + len(lst_ibm) + len(lst_ibm_noisy))
    val_ind_noisy += len(lst)
    val_ind_ibm += (len(lst) + len(lst_noisy))
    val_ind_ibm_noisy += (len(lst) + len(lst_noisy) + len(lst_ibm)) 
    val_ind_ms += (len(lst) + len(lst_noisy) + len(lst_ibm) + len(lst_ibm_noisy))
    val_ind_ms_noisy += (len(lst) + len(lst_noisy) + len(lst_ibm) + len(lst_ibm_noisy) + len(lst_ms))


    # choose whether to include noisy data
    train_ind = np.concatenate((train_ind, train_ind_noisy, train_ind_ibm, train_ind_ibm_noisy, train_ind_ms, train_ind_ms_noisy))
    # val_ind = np.concatenate((val_ind, val_ind_noisy, val_ind_ibm, val_ind_ibm_noisy, val_ind_ms, val_ind_ms_noisy))
    # train_ind = np.concatenate((train_ind, train_ind_ibm, train_ind_ms))
    val_ind = np.concatenate((val_ind, val_ind_ibm, val_ind_ms))
    val_ind = val_ind.astype(int)
    # train_ind = np.concatenate((train_ind, train_ind_ibm))
    # val_ind = np.concatenate((val_ind, val_ind_ibm))


    data_arr_train = data_arr[train_ind]
    data_arr_val = data_arr[val_ind]

    label_vecs_train = label_vecs[train_ind]
    label_vecs_val = label_vecs[val_ind]

    seq_lens_train = seq_lens[train_ind]
    seq_lens_val = seq_lens[val_ind]


    # Normalize data
    # NOTE: Be careful while subtracting mean!! Should not be subtracted from padded regions (else masking won't work)
    # mean, std = get_mean_std(data_arr_train, seq_lens_train)
    # for i in range(data_arr_train.shape[0]):
    #     data_arr_train[i,:seq_lens_train[i],:] = data_arr_train[i,:seq_lens_train[i],:] - mean
    # data_arr_train = data_arr_train / std
    # for i in range(data_arr_val.shape[0]):
    #     data_arr_val[i,:seq_lens_val[i],:] = data_arr_val[i,:seq_lens_val[i],:] - mean
    # data_arr_val = data_arr_val / std


    return data_arr_train, data_arr_val, label_vecs_train, label_vecs_val
    


if __name__ == '__main__':

    start = timeit.default_timer()

     
    model = WordNet()("train")
    # model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5), 
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])

    model.load_weights('/home/data1/anshulg/models/model_sepspk576_mix_bn_true_drop_final.h5', by_name=True) 
    # model.load_weights('/home/data1/anshulg/triplet_relu_newdata_deep2_drop_newdata.keras', by_name=True)
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

        

    filepath = "/home/data1/anshulg/models/model_sepspk576_newdata_2048_noisy.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]


    train_size = len(data_arr_train)
    val_size = len(data_arr_val)


    history = model.fit_generator(generator(data_arr_train, label_vecs_train), steps_per_epoch=int(train_size/BATCH_SIZE), epochs=40, callbacks=callbacks_list, validation_data=generator(data_arr_val, label_vecs_val), validation_steps=int(val_size/BATCH_SIZE))


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("/home/anshulg/plots/model_loss_sepspk576_newdata_2048_noisy.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig("/home/anshulg/plots/model_acc_sepspk576_2048_noisy.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    with open('/home/data1/anshulg/histories/history_sepspk576_2048_noisy.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    time = open('time.txt','w')
    time.write(str(stop-start))