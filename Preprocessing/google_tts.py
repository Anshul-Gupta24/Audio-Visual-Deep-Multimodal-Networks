import pandas as pd
import numpy as np
import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/home/anshul/Downloads/google_cloud_tts.json"


#df = pd.read_csv('experiment_labels.csv', encoding='utf8')

#katas = df['kata'].values
#num_katakana = np.sum(katas=='katakana')
#num_hiragana = np.sum(katas=='hiragana')
#print(num_katakana)
#print(num_hiragana)
#input()


#katakana = df['jap'].values[katas=='katakana']
#eng_katakana = df['eng'].values[katas=='katakana']
#hiragana = df['jap'].values[katas=='hiragana']
#eng_hiragana = df['eng'].values[katas=='hiragana']

#a = u"ja-JP"
#v = u"ja-JP-Wavenet-A"
#c = 0

#done = os.listdir('./synth_jap2_txt')
#for i in range(len(katakana)):
#    if 'synthesize-text_'+eng_katakana[i]+'_0.txt' not in done:
#        print(eng_katakana[i])
#        os.system(u' '.join(("bash get_tts.sh", katakana[i], eng_katakana[i], a, v, str(c))).encode('utf-8'))

#for i in range(len(hiragana)):
#    if 'synthesize-text_'+eng_hiragana[i]+'_0.txt' not in done:
#        print(eng_hiragana[i])
#        os.system(u' '.join(("bash get_tts.sh", hiragana[i], eng_hiragana[i], a, v, str(c))).encode('utf-8'))



#words = ['screen', 'map', 'ship', 'crown', 'swan', 'shoes', 'socks']
words = open('labels_exp2.txt','r').read().split('\n')
words = words[:-1]
#eng = df['eng'].values
#words_inds = [i for i in range(len(eng)) if eng[i] in words]
#jap = df['jap'].values
#for i in words_inds:
#        print(eng[i])
#        os.system(u' '.join(("bash get_tts.sh", jap[i], eng[i], a, v, str(c))).encode('utf-8'))


accents = ['en-gb', 'en-au','en-us']
voices = {}
voices['en-gb'] = ['en-GB-Wavenet-A','en-GB-Wavenet-B','en-GB-Wavenet-C','en-GB-Wavenet-D']
voices['en-au'] = ['en-AU-Wavenet-A','en-AU-Wavenet-B','en-AU-Wavenet-C','en-AU-Wavenet-D']
voices['en-us'] = ['en-US-Wavenet-A','en-US-Wavenet-B','en-US-Wavenet-C','en-US-Wavenet-D','en-US-Wavenet-E','en-US-Wavenet-F']

for w in words:
	c = 0
	for a in accents:
		for v in voices[a]:
			os.system(u' '.join(("bash get_tts_eng.sh", w, a, v, str(c))).encode('utf-8'))
			c += 1
