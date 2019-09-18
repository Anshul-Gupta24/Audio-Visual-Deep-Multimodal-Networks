'''
Code to get image and audio retrieval accuracies
'''
import numpy as np
np.random.seed(1337)
import pickle
import os
from model_proxy import JointNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'



input_size = 80
hidden_size = 128
NUM_CLASSES = 655


def get_model(filepath):

    jointnet = JointNet(np.zeros((576, NUM_CLASSES)))
    model = jointnet.model
    aud_transform = jointnet.audio_submodel
    img_transform = jointnet.image_submodel

    model.load_weights(filepath, by_name=True)
    return img_transform, aud_transform

    
def get_confusion(speech_data, img_data_val, img_transform, aud_transform):

    classes = open('classes.txt').read().split('\n')
    classes = classes[:-1]

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_test.pkl', 'rb') as fp:
        spk_ibm_test = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_test.pkl', 'rb') as fp:
        spk_google_test = pickle.load(fp)

    with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_test.pkl', 'rb') as fp:
        spk_ms_test = pickle.load(fp)

    spk_ibm_test += 14
    spk_ms_test += 14 + 3
    spk_test = np.concatenate((spk_google_test, spk_ibm_test, spk_ms_test))

    aud_features = []
    for c in classes:
        for s in spk_test:
            aud_features.append(speech_data[c][s])

    aud_latent = aud_transform.predict(np.array(aud_features))
   
    start = 16
    end = 32
    img_features = []
    for c in classes:
        for s in range(start, end):
            img_features.append(img_data_val[c][s])

    img_latent = img_transform.predict(np.array(img_features))
    

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
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            for y, ci in enumerate(classes):
                img_ind = np.random.randint(num_images)
                v2 = img_latent[y*num_images + img_ind]                
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                
                cmat[x][y] = np.dot(v1, v2)
        
        folder = 'Confusions/confusion_proxy_audio_anchor/'
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
        with open('Confusions/confusion_proxy_audio_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
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

        folder = 'Confusions/confusion_proxy_image_anchor/'
        if not os.path.isdir(folder):
            os.system('mkdir '+folder)
        with open('Confusions/confusion_proxy_image_anchor/confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)


def top_k(cmat, k):
    
    acc = 0
    for x in range(NUM_CLASSES):
        topk_inds = np.argsort(cmat[x,:])[-k:]
        if x in topk_inds:
            acc += 1

    acc = acc/NUM_CLASSES
    return acc

    
def accuracy(folder):
    
    acc_top1 = 0
    acc_top5 = 0
    for i in range(20):
        with open(folder + 'confusion_mat_' + str(i) + '.pkl', 'rb') as fp:
            cmat = pickle.load(fp)
        
        acc_top1 += top_k(cmat, 1)
        acc_top5 += top_k(cmat, 5)
        
    acc_top1 = acc_top1/20
    acc_top5 = acc_top5/20
    
    return acc_top1, acc_top5
            
            

if __name__=='__main__':

    with open('Data/img_data_val.pkl', 'rb') as fp:
        img_data_val = pickle.load(fp)

    with open('/home/data1/anshulg/speech_features_2048D.pkl', 'rb') as fp:
        speech_data = pickle.load(fp) 

    filepath = 'Saved_models/saved-model-125.hdf5'      # choose model to load
    img_transform, aud_transform = get_model(filepath)
    get_confusion(speech_data, img_data_val, img_transform, aud_transform)
    print('Image retrieval accuracy:')
    top1, top5 = accuracy('Confusions/confusion_proxy_audio_anchor/')
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    print('Audio retrieval accuracy:')
    top1, top5 = accuracy('Confusions/confusion_proxy_image_anchor/')
    print('Top 1: ' + str(top1))
    print('Top 5: ' + str(top5))
    
