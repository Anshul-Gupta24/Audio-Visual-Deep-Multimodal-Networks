'''
Creates the train, val and test image data files:
Data/image_features_train.pkl
Data/image_features_val.pkl
Data/image_features_test.pkl
'''
import numpy as np
import pickle


classes = open('../classes.txt','r').read().split('\n')[:-1]
new_classes = open('../new_classes.txt','r').read().split('\n')[:-1]
for c in new_classes:
  if c in classes:
    new_classes.remove(c)
    
with open('../Data/image_features_imagenet_train.pkl', 'rb') as fp:
  image_features_imagenet_train = pickle.load(fp)
with open('../Data/image_features_imagenet_val.pkl', 'rb') as fp:
  image_features_imagenet_val = pickle.load(fp)
with open('../Data/image_features_google.pkl', 'rb') as fp:
  image_features_google = pickle.load(fp)
with open('../Data/image_features_flickr.pkl', 'rb') as fp:
  image_features_flickr = pickle.load(fp)
  
    
# create train pkl file
img_features_train = {}
for c in classes:
  img_features_train[c] = image_features_imagenet_train[c][:160]
  
for c in new_classes:
  img_features_train[c] = image_features_google[c][:30]
  img_features_train[c] = np.concatenate(img_features_train[c], image_features_flickr[c][:30])
  
  
# create val pkl file
img_features_val = {}
for c in classes:
  img_features_val[c] = image_features_imagenet_val[c][:16]
  
for c in new_classes:
  img_features_val[c] = image_features_google[c][30:40]
  img_features_val[c] = np.concatenate(img_features_val[c], image_features_flickr[c][30:40])
  
  
# create test pkl file
img_features_test = {}
for c in classes:
  img_features_test[c] = image_features_imagenet_val[c][16:32]
  
for c in new_classes:
  img_features_test[c] = image_features_google[c][40:50]
  img_features_test[c] = np.concatenate(img_features_test[c], image_features_flickr[c][40:50])


with open('../Data/image_features_train.pkl', 'wb') as fp:
  pickle.dump(img_features_train, fp)
with open('../Data/image_features_val.pkl', 'wb') as fp:
  pickle.dump(img_features_val, fp)
with open('../Data/image_features_test.pkl', 'wb') as fp:
  pickle.dump(img_features_test, fp) 
