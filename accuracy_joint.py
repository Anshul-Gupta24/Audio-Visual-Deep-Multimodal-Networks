import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation, InputLayer
import numpy as np
# np.random.seed(1337)
import pickle
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from model_proxy_eng import JointNet
from model import WordNet



import sys
sys.path.insert(0, '/home/anshulg/WordNet/get_imagenet/functions')
# import imagenet_posterior



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['OMP_NUM_THREADS']='8'



input_size = 80
hidden_size = 128
NUM_CLASSES = 576


def get_model(filepath):

    ####################

    # TRIPLET

    ####################


    # aud_transform = Sequential(name='sequential_1')
    # # # aud_transform.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))		
    # # # aud_transform.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=False))
    # # # aud_transform.add(LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=False))
    # # # aud_transform.add(Dense(100, name='dense_1', trainable=False))
    # # # aud_transform.add(BatchNormalization(name='batch_normalization_1', trainable=False))
    # # # aud_transform.add(Activation('relu'))
    # # # aud_transform.add(Dropout(0.2))

    # # # aud_transform.load_weights('/home/data1/anshulg/models/model_sepspk576_mix_bn_true_drop_final.h5', by_name=True)

    # # aud_transform.add(Dense(1024, activation='relu', input_shape=(100,), name='sequential_1/dense_aud0'))
    # # aud_transform.add(Dense(512, activation='relu', name='sequential_1/dense_aud1'))
    # aud_transform.add(Dense(576, activation='sigmoid', input_shape=(2048,), name='dense_aud1'))
    # # aud_transform.add(Dropout(0.2))
    # # aud_transform.add(BatchNormalization(name='batchnorm_1'))
    # # aud_transform.add(Activation('relu'))
    # # aud_transform.add(Dense(128, activation='relu', name='dense_aud2'))
    # # aud_transform.add(Dense(64, activation='relu', name='dense_aud0'))

    # img_transform = Sequential(name='sequential_2')
    # # img_transform.add(Dense(1024, activation='relu', input_shape=(4096,), name='sequential_2/dense_img0'))
    # # img_transform.add(Dense(512, activation='relu', name='sequential_2/dense_img1'))
    # img_transform.add(Dense(576, activation='sigmoid', input_shape=(2048,), name='dense_img1'))
    # # img_transform.add(Dropout(0.2))
    # # aud_transform.add(BatchNormalization(name='batchnorm_2'))
    # # aud_transform.add(Activation('relu'))
    # # img_transform.add(Dense(128, activation='relu', name='dense_img2'))
    # # img_transform.add(Dense(64, activation='relu', name='dense_img0'))
    # aud_transform.load_weights('/home/data1/anshulg/triplet_relu_newdata_deep2_2048_semihard576_sigmoid.keras', by_name=True)
    # img_transform.load_weights('/home/data1/anshulg/triplet_relu_newdata_deep2_2048_semihard576_sigmoid.keras', by_name=True)




    #####################

    #   PROXY

    #####################


    jointnet = JointNet(np.zeros((576, NUM_CLASSES)))
    model = jointnet.model
    aud_transform = jointnet.audio_submodel
    img_transform = jointnet.image_submodel
    #img_transform = []

    print(aud_transform.layers[-1].get_weights())
    input()

    model.load_weights(filepath, by_name=True)
   
    print(aud_transform.summary())
    print(img_transform.summary())
    print(aud_transform.layers[-1].get_weights())
    input()



    #####################

    # MODEL CORRELATION

    #####################


    # aud_transform = Sequential()
    # aud_transform.add(InputLayer((2048, )))
    # aud_transform.add(Dense(576, activation='softmax'))
    # with open('/home/anshulg/WordNet/proxy_data/speech_weights.pkl', 'rb') as fp:
    #         weights = pickle.load(fp)
    # aud_transform.layers[-1].set_weights(weights)


    # img_transform = Sequential()
    # img_transform.add(InputLayer((2048, )))
    # img_transform.add(Dense(576, activation='softmax'))  
    # with open('/home/anshulg/WordNet/proxy_data/xception_weights.pkl', 'rb') as fp:
    #         weights = pickle.load(fp)
    # img_transform.layers[-1].set_weights(weights)


    # print(aud_transform.layers[-1].get_weights())
    # input()


    return img_transform, aud_transform




def get_distance(img_features, aud_features, img_transform, aud_transform):

    # classes = open('/home/anshulg/WordNet/labels_removed2.txt').read().split('\n')
    # classes = classes[:-1]

    # xinds = imagenet_posterior.xception_inds(classes)


    st = time.time()
    img_latent = img_transform.predict(img_features)
    en = time.time()
    print(en - st)

    st = time.time()
    img_latent = img_latent / (np.expand_dims(np.linalg.norm(img_latent, axis=1), axis=1) + 1e-8)
    en = time.time()
    print(en - st)

    st = time.time()
    print(aud_features[0].shape)
    aud_latent = aud_transform.predict(aud_features)
    en = time.time()
    print(en - st)

    st = time.time()
    aud_latent = aud_latent / (np.expand_dims(np.linalg.norm(aud_latent, axis=1), axis=1) + 1e-8)
    en = time.time()
    print(en - st)

    ####Kiran####
    # img_latent_norm = np.linalg.norm(img_latent, axis=-1) + 1e-8
    # img_latent = (img_latent.T / img_latent_norm).T

    # aud_latent_norm = np.linalg.norm(aud_latent, axis=-1) + 1e-8
    # aud_latent = (aud_latent.T / aud_latent_norm).T
    #############

    dist = np.sum(img_latent*aud_latent, axis=1)
    # dist = np.maximum(np.zeros(img_features.shape[0]), np.sum(img_latent*aud_latent, axis=1))

    return dist


# NOTE: 1 implies similar pair, 0 implies dissimilar pair
def get_prediction(dist, thres):

    pred = np.zeros(len(dist))
    for i,d in enumerate(dist):
        if d>thres:
            pred[i]=1
        else:
            pred[i]=0

    return pred


def accuracy_speechclass(spclass, csv_path, speech_data, img_data_val):

    df = pd.read_csv(csv_path)
    groundings = df['0'].values
    pos_img_labels = df['1'].values
    pos_img_indices = df['2'].values
    pos_aud_labels = df['3'].values
    pos_aud_indices = df['4'].values
    neg_labels = df['5'].values
    neg_indices = df['6'].values


    img_features = []
    aud_features = []
    y_true = []

    shift = 16
    for i in range(df.shape[0]):
        if groundings[i] == 1:
            if pos_aud_labels[i] == spclass:
                img_features.append(img_data_val[pos_img_labels[i]][pos_img_indices[i]] + shift)
                aud_features.append(speech_data[pos_aud_labels[i]][pos_aud_indices[i]])
                y_true.append(1) 
                img_features.append(img_data_val[neg_labels[i]][neg_indices[i]] + shift)
                aud_features.append(speech_data[pos_aud_labels[i]][pos_aud_indices[i]])
                y_true.append(0)                

    y_true = np.array(y_true)
    img_features = np.array(img_features)
    aud_features = np.array(aud_features)

    dist = get_distance(img_features, aud_features)
    y_pred = get_prediction(dist)

    num_samples = len(y_pred)
    print()
    print()
    print(spclass)
    print('num_samples: '+str(num_samples))
    accuracy = np.sum(y_pred==y_true) / num_samples

    return accuracy 


def accuracy_imageclass(csv_path, speech_data, img_data_val, img_transform, aud_transform, imclass=None):

    df = pd.read_csv(csv_path)
    groundings = df['0'].values
    pos_img_labels = df['1'].values
    pos_img_indices = df['2'].values
    pos_aud_labels = df['3'].values
    pos_aud_indices = df['4'].values
    neg_labels = df['5'].values
    neg_indices = df['6'].values


    img_features = []
    aud_features = []
    y_true = []
    pairs = []

    for i in range(df.shape[0]):
    # for i in range(1000):
        if groundings[i] == 0:
            if(pos_img_labels[i] == imclass or imclass == None):
                img_features.append(img_data_val[pos_img_labels[i]][pos_img_indices[i]])
                aud_features.append(speech_data[pos_aud_labels[i]][pos_aud_indices[i]])
                pairs.append((pos_img_labels[i], pos_aud_labels[i]))
                y_true.append(1) 
                img_features.append(img_data_val[pos_img_labels[i]][pos_img_indices[i]])
                aud_features.append(speech_data[neg_labels[i]][neg_indices[i]])
                pairs.append((pos_img_labels[i], neg_labels[i]))
                y_true.append(0)                

    y_true = np.array(y_true)
    img_features = np.array(img_features)
    aud_features = np.array(aud_features)

    dist = get_distance(img_features, aud_features, img_transform, aud_transform)
    print(dist)
    # for i, p in enumerate(pairs):
    #     confusion_mat[class_to_ind[p[0]]][class_to_ind[p[1]]] += dist[i]
    #     count_mat[class_to_ind[p[0]]][class_to_ind[p[1]]] += 1


    # count_mat += 0.000001
    # confusion_mat = confusion_mat / count_mat


    # with open('confusion_mat_imagenet_randneg.pkl','wb') as fp:
    #     pickle.dump(confusion_mat, fp)

    max_accuracy = 0
    max_thres = 0
    # thresholds = np.linspace(0,1,10,endpoint=False)
    thresholds = [0.2]
    for thres in thresholds:
        y_pred = get_prediction(dist, thres)

        num_samples = len(y_pred)
        print()
        print()
        print('Class: ', imclass)
        print('Threshold: ', thres)
        print('num_samples: '+str(num_samples))
        accuracy = np.sum(y_pred==y_true) / num_samples
        print('Accuracy: ', accuracy)
        if accuracy>max_accuracy:
            max_accuracy = accuracy
            max_thres = thres

        # check = [(dist[i],y_pred[i],y_true[i]) for i in range(num_samples)]
        # print(check)


    return max_thres, max_accuracy 


def accuracy_all(csv_path, speech_data, img_data_val):

    df = pd.read_csv(csv_path)
    img_labels = df['0'].values
    img_indices = df['1'].values
    aud_labels = df['2'].values
    aud_indices = df['3'].values

    
    count_mat = np.zeros((576, 576))


    # with open('/home/anshulg/classes_imagenet.pkl', 'rb') as fp:
    #     classes = pickle.load(fp)

    classes = open('/home/anshulg/WordNet/labels_removed2.txt').read().split('\n')
    classes = classes[:-1]

    class_to_ind = {c:i for i,c in enumerate(classes)}
    # with open('confusion_class_ind.pkl','wb') as fp:
    #     pickle.dump(class_to_ind, fp)

    start = 0
    fac = int(576*576 / 20)
    end = start + fac
    for num_mat in range(20):
        confusion_mat = np.zeros((576, 576))

        print()
        print('confusion mat: ',num_mat) 
        print()

        for x in range(21):

            # print(start)
            # print(end)
            print(x)

            img_features = []
            aud_features = []
            y_true = []
            pairs = []

            for i in range(start, end):
                img_features.append(img_data_val[img_labels[i]][img_indices[i]])
                aud_features.append(speech_data[aud_labels[i]][aud_indices[i]])
                pairs.append((img_labels[i], aud_labels[i]))
                
                if(img_labels[i]==aud_labels[i]):
                    y_true.append(1)
                else:
                    y_true.append(0)


            y_true = np.array(y_true)
            img_features = np.array(img_features)
            aud_features = np.array(aud_features)

            print('start')
            dist = get_distance(img_features, aud_features)
            print('end')
            # input()

            for i, p in enumerate(pairs):
                confusion_mat[class_to_ind[p[0]]][class_to_ind[p[1]]] = dist[i]
                # count_mat[class_to_ind[p[0]]][class_to_ind[p[1]]] += 1

            start = end
            end = start + fac
            if(x==19):
                end = (num_mat+1)*576*576


        with open('/home/anshulg/joint_confusion_old/confusion_mat_'+str(num_mat)+'.pkl','wb') as fp:
            pickle.dump(confusion_mat, fp)
        print('done')


    # count_mat += 0.000001
    # confusion_mat = confusion_mat / count_mat


    

    # max_accuracy = 0
    # max_thres = 0
    # # thresholds = np.linspace(0,1,10,endpoint=False)
    # thresholds = [0.5]
    # for thres in thresholds:
    #     y_pred = get_prediction(dist, thres)

    #     num_samples = len(y_pred)
    #     print()
    #     print()
    #     # print('Class: ', imclass)
    #     print('Threshold: ', thres)
    #     print('num_samples: '+str(num_samples))
    #     accuracy = np.sum(y_pred==y_true) / num_samples
    #     print('Accuracy: ', accuracy)
    #     if accuracy>max_accuracy:
    #         max_accuracy = accuracy
    #         max_thres = thres

    #     # check = [(dist[i],y_pred[i],y_true[i]) for i in range(num_samples)]
    #     # print(check)


    # return max_thres, max_accuracy 



def accuracy_all2(speech_data, img_data_val, img_transform, aud_transform, epoch_num):

    classes = open('/home/anshulg/WordNet/labels_removed2.txt').read().split('\n')
    classes = classes[:-1]

    # classes = open('/home/anshulg/WordNet/labels_exp2.txt','r').read().split('\n')
    # classes = classes[:-1]

    NUM_CLASSES = len(classes)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_test.pkl', 'rb') as fp:
        spk_ibm_test = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_test.pkl', 'rb') as fp:
        spk_google_test = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_test.pkl', 'rb') as fp:
        spk_ms_test = pickle.load(fp)

    spk_ibm_test += 14
    spk_ms_test += 14 + 3
    spk_test = np.concatenate((spk_google_test, spk_ibm_test, spk_ms_test))

    # spk_google_test = np.array([])
    # spk_ibm_test = np.array([])
    # spk_ms_test = np.array([2])
    # spk_ibm_test += 1
    # spk_ms_test += 1 +1
    # spk_test = np.concatenate((spk_google_test, spk_ibm_test, spk_ms_test))
    # spk_test = spk_test.astype(int)

    aud_features = []
    for c in classes:
        for s in spk_test:
            aud_features.append(speech_data[c][s])


    aud_latent = aud_transform.predict(np.array(aud_features))
    # aud_latent = aud_latent / (np.expand_dims(np.linalg.norm(aud_latent, axis=1), axis=1) + 1e-8)

    start = 16
    end = 32
    img_features = []
    for c in classes:
        for s in range(start, end):
            img_features.append(img_data_val[c][s])


    img_latent = img_transform.predict(np.array(img_features))
    # img_latent = img_latent / (np.expand_dims(np.linalg.norm(img_latent, axis=1), axis=1) + 1e-8)


    # xinds = imagenet_posterior.xception_inds(classes)


    # Get audio anchor confusions


    num_speakers = len(spk_test)
    num_images = end - start
    print('audio anchor')
    for i in range(20):
        print(i)
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for x, ca in enumerate(classes):
            spk_ind = np.random.randint(num_speakers)
            v1 = aud_latent[x*num_speakers + spk_ind]
            # print(v1)
            # input()
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            for y, ci in enumerate(classes):
                img_ind = np.random.randint(num_images)
                v2 = img_latent[y*num_images + img_ind]
                
                # if ci==ca:
                #     print(v2)
                #     input()
                
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                cmat[x][y] = np.dot(v1, v2)
                # cmat[x][y] = np.sum(np.power(aud_latent[x*5 + spk_ind] - img_latent[y*16 + img_ind], 2))
        
        # folder = '/home/anshulg/joint_confusions/joint_confusion_proxy_softmax_jap_audio_anchor/epoch_'+str(epoch_num)
        # if not os.path.isdir(folder):
        #     os.system('mkdir '+folder)
        with open('/home/anshulg/joint_confusions/joint_confusion_proxy_softmax_jap_audio_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)

    
    # Get image anchor confusions

    print('image anchor')
    for i in range(20):
        print(i)
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for x, ci in enumerate(classes):
            img_ind = np.random.randint(num_images)
            v1 = img_latent[x*num_images + img_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            for y, ca in enumerate(classes):
                spk_ind = np.random.randint(num_speakers)
                v2 = aud_latent[y*num_speakers + spk_ind]
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                cmat[x][y] = np.dot(v1, v2)
                # cmat[x][y] = np.sum(np.power(aud_latent[y*5 + spk_ind] - img_latent[x*16 + img_ind], 2))
        # folder = '/home/anshulg/joint_confusions/joint_confusion_proxy_softmax_jap_image_anchor/epoch_'+str(epoch_num)
        # if not os.path.isdir(folder):
        #     os.system('mkdir '+folder)
        with open('/home/anshulg/joint_confusions/joint_confusion_proxy_softmax_jap_image_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)




def get_confusion_csv():

    with open('/home/anshulg/confusion_class_ind.pkl', 'rb') as fp:
        cind = pickle.load(fp)

    with open('/home/anshulg/confusion_mat.pkl', 'rb') as fp:
        cmat = pickle.load(fp)


    sorted_ind = sorted(cind.items(), key=lambda kv: kv[1])
    classes = [s[0] for s in sorted_ind]

    df = pd.DataFrame()
    df['0'] = classes
    for c in classes:
        df[c] = cmat[:][cind[c]]

    df.to_csv('confusion_mat.csv')


def get_confusion_plot():

    with open('/home/anshulg/confusion_mat_imagenet_randneg.pkl', 'rb') as fp:
        cmat = pickle.load(fp)

    plt.imshow(cmat)
    plt.savefig('/home/anshulg/confusion_mat_imagenet_randneg.png')
    plt.close()



def accuracy_jap(img_data):

    with open('/home/data1/anshulg/google_synth_features_jap.pkl', 'rb') as fp:
        data = pickle.load(fp)

    lst = list(data.keys())
    labels = np.array([l.split('_')[-2] for l in lst])


    df = pd.read_csv('/home/anshulg/WordNet/get_imagenet/functions/imagenet-katakana.csv')
    katas = df['kata'].values
    katakana = df['eng'].values[katas=='katakana']
    katakana_inds = [i for i in range(len(labels)) if labels[i] in katakana]
    print(katakana_inds)
    labels_katakana = labels[katakana_inds]

    with open('/home/anshulg/classes_imagenet.pkl', 'rb') as fp:
        classes = pickle.load(fp)

    ind_to_class = {i:c for i,c in enumerate(classes)}


    accuracy = 0
    for i in katakana_inds:
        print(labels[i])
        img_features = []
        aud_features = []

        aud_features = [data[lst[i]]] * 576
        for c in classes:
            img_features.append(img_data[c][0])
        

        img_features = np.array(img_features)
        aud_features = np.array(aud_features)
        dist = get_distance(img_features, aud_features)
        pred_label_ind = np.argmax(dist)
        pred_top5_ind = dist.argsort()[-5:][::-1]
        pred_label = ind_to_class[pred_label_ind]
        pred_top5 = [ind_to_class[t] for t in pred_top5_ind]

        print('pred labels:', pred_top5)
        if(pred_label==labels[i]):
            print('yes!!')
            accuracy += 1


    accuracy = accuracy / len(katakana_inds)

    return accuracy



if __name__=='__main__':

    # with open('/home/data1/anshulg/ImageNet/data_val.pkl', 'rb') as fp:
    with open('/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_val_3x.pkl', 'rb') as fp:
        img_data_val = pickle.load(fp)

    with open('/home/data1/kiranp/Flickr_selected/feats_test.pkl', 'rb') as fp:
        flickr_test = pickle.load(fp)

    # with open('/home/data1/kiranp/Google_images_selected/feats_test.pkl', 'rb') as fp:
    #     google_test = pickle.load(fp)


    with open('/home/data1/anshulg/speech_features_2048D.pkl', 'rb') as fp:
        speech_data = pickle.load(fp) 

    csv_path = '/home/anshulg/WordNet/get_imagenet/test_data_mix2.csv'

    # Class accuracy
    # imclass = 'vestment'
    # _, accuracy = accuracy_imageclass(csv_path, speech_data, img_data_val, imclass)
    # print(imclass+' accuracy: '+str(accuracy))

    # # Total accuracy
    # max_thres, accuracy = accuracy_imageclass(csv_path, speech_data, img_data_val)
    # print()
    # print()
    # print('Total accuracy: '+str(accuracy))
    # print('Best threshold: '+str(max_thres))

    # folder = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu3'
    # files = os.listdir(folder)
    # epoch = sorted([f.split('-')[-1][:-5] for f in files])
    # for i in range(5, 30):
    #     print('EPOCH: ', i)
    #     print()
    #     print()
    #     filepath = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu3/saved-model-'+str(epoch[i])+'.hdf5'
    #     aud_transform = get_model(filepath)
    #     accuracy_all2(speech_data, img_data_val, aud_transform, aud_transform, i)

    # filepath = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap/saved-model-119.hdf5'
    filepath = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4/saved-model-25.hdf5'
    # filepath = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap/saved-model-84.hdf5'
    # filepath = '/home/data1/anshulg/proxy2_loss2_deep_softmax_relu4_jap2/saved-model-42.hdf5'
    img_transform, aud_transform = get_model(filepath)
    # accuracy_all2(speech_data, img_data_val, img_transform, aud_transform, 0)
    max_thres, accuracy = accuracy_imageclass(csv_path, speech_data, img_data_val, img_transform, aud_transform)
    print('Total accuracy: '+str(accuracy))

    # print(accuracy_jap(img_data_val))
