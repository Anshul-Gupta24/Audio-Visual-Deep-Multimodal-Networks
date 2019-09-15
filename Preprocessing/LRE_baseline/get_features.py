#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:53:55 2019

@author: shreyasr
"""

import os
import textwrap
import argparse as ap
import configparser as cp
import numpy as np
from sklearn.svm import SVC
from lre_system import LRESystem as LRE17System
import lib.backend as bk
from lib.utils import h5write, h5read, mkdir_p
import sys
import pickle


config = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
try:
    config.read('lre17_bnf_baseline.cfg')
except:
    raise IOError('Something is wrong with the config file.')
    

lre17_system = LRE17System(config)
base_path = '../Data/
locs = ['google_synth_eng', 'ibm_synth_eng', 'microsoft_synth_eng', 'google_synth_jap', 'ibm_synth_jap', 'microsoft_synth_jap']
pkls = ['google_synth_features.pkl', 'ibm_synth_features.pkl', 'microsoft_synth_features.pkl', 'google_synth_features_jap.pkl', 'ibm_synth_features_jap.pkl', 'microsoft_synth_features_jap.pkl'] 
features = {}

for j, loc in enumerate(locs):
	for i, w in enumerate(os.listdir(base_path+loc)):
		if(w[-3:]=='wav'):
			print(w)
			print(i)
			feat = lre17_system.extract_feat_and_apply_sad_then_cmvn(path+'/'+w)
			features[w] = feat.T


	with open(base_path + pkls[j],'wb') as fp:
		pickle.dump(features,fp)
	
