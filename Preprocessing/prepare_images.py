import numpy as np
import pickle
from functions.data import im_load
from keras.applications.xception import preprocess_input
from functions.model import get_image_model_xception
import os

def is_folder(name):
    return len(name.split('.')) == 1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# base_path = '/home/data1/kiranp/Flickr_selected'
base_path = '/home/data1/kiranp/Google_images_selected'
# base_path = 'Flickr_selected'
folders = sorted([item for item in os.listdir(base_path) if is_folder(item)])

image_feats = {}

# with open(base_path + '/feats.pkl', 'rb') as file:
#     image_feats = pickle.load(file)
# folders = ['thorns']

for class_name in folders:
    print('Starting {0:s}...({1:d})'.format(class_name, len(os.listdir('{0:s}/{1:s}'.format(base_path, class_name)))))
    filepaths = np.array(['{0:s}/{1:s}/{2:s}'.format(base_path, class_name, item) for item in os.listdir('{0:s}/{1:s}'.format(base_path, class_name))])
    batch_size = len(filepaths)
    image_model = get_image_model_xception()
    image_batch = np.zeros((batch_size, 224, 224, 3))
    for idx in range(0, batch_size):
        print(' {0:d}/{1:d}'.format(idx+1, len(image_batch)), end='\r')
        try:
            image_batch[idx] = im_load(filepaths[idx])
        except:
            pass
    print('Preprocessing')
    preprocess_input(image_batch)
    print('Predicting')
    image_feat = image_model.predict(image_batch, batch_size=50)
    print(image_feat.shape)
    image_feats[class_name] = image_feat

with open('{0:s}/feats.pkl'.format(base_path), 'wb') as file:
    pickle.dump(image_feats, file)
