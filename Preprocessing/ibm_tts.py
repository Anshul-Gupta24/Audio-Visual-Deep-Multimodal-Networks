# -*- coding: utf-8 -*-
'''
Script to generate IBM TTS data
'''
import os
import numpy as np
import pandas as pd


words = open('../classes.txt','r').read().split('\n')
words = words[:-1]

# English voices
voices = ['en-US_AllisonVoice', 'en-US_MichaelVoice', 'en-GB_KateVoice']
# Japanese voices
#voices = ['ja-JP_EmiVoice']

for w in words:
	c = 0
	for v in voices:
		print(w)
		print()	
		os.system(("get_ibm_tts.sh"+" "+w+" "+v+" _"+str(c)).encode('utf-8'))
		c += 1
