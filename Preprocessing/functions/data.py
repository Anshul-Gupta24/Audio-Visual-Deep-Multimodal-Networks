import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array


def im_load(filename, target_size=(224, 224)):
    img = load_img(filename, target_size=target_size)
    img = img_to_array(img)
    return img

def get_label_to_remap():
    df = pd.read_csv('labels_remap.csv')
    labels = df.values[:, 0]
    remap = df.values[:, 1]
    l2r = dict(zip(labels, remap))
    return l2r
