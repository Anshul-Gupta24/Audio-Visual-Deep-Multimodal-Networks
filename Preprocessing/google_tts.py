'''
Script to generate Google TTS data
'''
import pandas as pd
import numpy as np
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="<location to json file>"


words = open('../classes.txt','r').read().split('\n')
words = words[:-1]

# English voices
accents = ['en-gb', 'en-au','en-us']
voices = {}
voices['en-gb'] = ['en-GB-Wavenet-A','en-GB-Wavenet-B','en-GB-Wavenet-C','en-GB-Wavenet-D']
voices['en-au'] = ['en-AU-Wavenet-A','en-AU-Wavenet-B','en-AU-Wavenet-C','en-AU-Wavenet-D']
voices['en-us'] = ['en-US-Wavenet-A','en-US-Wavenet-B','en-US-Wavenet-C','en-US-Wavenet-D','en-US-Wavenet-E','en-US-Wavenet-F']
# Japanese voices
#accents = [u"ja-JP"]
#voices = {u"ja-JP":u"ja-JP-Wavenet-A"}


for w in words:
	c = 0
	for a in accents:
		for v in voices[a]:
			os.system(u' '.join(("bash get_google_tts.sh", w, a, v, str(c))).encode('utf-8'))
			c += 1
			
os.system("bash google_text_speech.sh")
