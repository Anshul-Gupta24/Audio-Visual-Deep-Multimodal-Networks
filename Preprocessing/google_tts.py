# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


# labels = open('/home/anshulg/WordNet/labels_removed2.txt','r').read().split('\n')
# labels = labels[:-1]

# df = pd.read_csv('/home/anshulg/WordNet/experiment_labels.csv')
# expt_labels = df['Label'].values

# labels = np.concatenate((labels, expt_labels))
# labels = set(labels)
# #print labels



# df = pd.read_csv('/home/anshulg/WordNet/experiment_labels.csv')
# katas = df['kata'].values
# katakana_eng = df['eng'].values[katas=='katakana']
# katakana = df['jap'].values[katas=='katakana']

# hiragana_eng = df['eng'].values[katas=='hiragana']
# hiragana = df['jap'].values[katas=='hiragana']

# eng = np.concatenate([katakana_eng, hiragana_eng])
# jap = np.concatenate([katakana, hiragana])
# print(jap)
# input()


# words = ['screen', 'map', 'ship', 'crown', 'swan', 'shoes', 'socks']
# words_inds = [i for i in range(len(eng)) if eng[i] in words]

# done = os.listdir('/home/data1/anshulg/ibm_synth_jap_exp')
# done = []
# for i, l in zip(words_inds, words):
# # for i, l in enumerate(eng):
# 	# l="umbrella"
# 	if l + '_0.wav' not in done:	
# 		print(l)
# 		print()	
# 		os.system(("bash /home/anshulg/WordNet/get_ibm_tts.sh"+" "+l+" "+jap[i]).encode('utf-8'))
# 		# input()


#os.system('bash get_ibm_tts.sh sup')


words = open('/home/anshulg/WordNet/labels_exp2.txt','r').read().split('\n')
words = words[:-1]

voices = ['en-US_AllisonVoice', 'en-US_MichaelVoice', 'en-GB_KateVoice']

for w in words:
	c = 0
	for v in voices:
		print(w)
		print()	
		os.system(("bash /home/anshulg/WordNet/get_ibm_tts.sh"+" "+w+" "+v+" _"+str(c)).encode('utf-8'))
		c += 1