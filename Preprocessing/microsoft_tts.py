import os
import numpy as np
import pandas as pd
from get_tts import TextToSpeech
import time
import pandas as pd



subscription_key = "27f7ef139ac44f049065ebdfe6f67762"
app = TextToSpeech(subscription_key)
app.get_token()


labels = open('/home/anshulg/WordNet/labels_removed2.txt','r').read().split('\n')
labels = labels[:-1]

#print labels


accents = ['en-US', 'en-IN','en-AU', 'en-CA', 'en-GB', 'en-IE']
voices = {}
voices['en-US'] = ['ZiraRUS', 'Jessa24KRUS', 'BenjaminRUS', 'Guy24KRUS']
voices['en-IN'] = ['Heera, Apollo', 'PriyaRUS', 'Ravi, Apollo']
voices['en-AU'] = ['Catherine', 'HayleyRUS']
voices['en-CA'] = ['Linda', 'HeatherRUS']
voices['en-GB'] = ['Susan, Apollo', 'HazelRUS', 'George, Apollo']
voices['en-IE'] = ['Sean']


# accents = ['ja-JP']
# voices = {}
# voices['ja-JP'] = ['Ayumi, Apollo', 'Ichiro, Apollo', 'HarukaRUS']



# df = pd.read_csv('/home/anshulg/WordNet/experiment_labels.csv')
# katas = df['kata'].values
# katakana_eng = df['eng'].values[katas=='katakana']
# katakana = df['jap'].values[katas=='katakana']

# hiragana_eng = df['eng'].values[katas=='hiragana']
# hiragana = df['jap'].values[katas=='hiragana']

# eng = np.concatenate([katakana_eng, hiragana_eng])
# jap = np.concatenate([katakana, hiragana])

# done = os.listdir('/home/data1/anshulg/microsoft_synth_jap_exp')
# count = 0
# for i in range(len(eng)):
#     if 'synthesize-audio_'+eng[i]+'_0.wav' not in done:
#         c = 0
#         print(eng[i])
#         for a in accents:
#             for v in voices[a]:      
#                 print(v)      
#                 print(c)
#                 app.save_audio(a, v, jap[i], eng[i], c)
#                 c += 1
#                 count +=1
#                 time.sleep(3)

#         if(count%195==0):
#             # time.sleep(5)
#             print()
#             print('new token!!!')
#             print()
#             app = TextToSpeech(subscription_key)
#             app.get_token()


#         print()

# words = ['screen', 'map', 'ship', 'crown', 'swan', 'shoes', 'socks']
# words_inds = [i for i in range(len(eng)) if eng[i] in words]

# for i in words_inds:
#     print(eng[i])
#     c = 0
#     for a in accents:
#         for v in voices[a]:  
#             print(v)      
#             print(c)
#             app.save_audio(a, v, jap[i], eng[i], c)
#             c += 1
#             time.sleep(3)

# words = open('/home/anshulg/WordNet/labels_exp2.txt','r').read().split('\n')
# words = words[:-1]
words = ['books', 'chair']


count = 0
for w in words:
    # if 'synthesize-audio_'+eng[i]+'_0.wav' not in done:
        c = 0
        print(w)
        for a in accents:
            for v in voices[a]:      
                print(v)      
                print(c)
                app.save_audio(a, v, '', w, c)
                c += 1
                count +=1
                time.sleep(3)

        if(count%195==0):
            # time.sleep(5)
            print()
            print('new token!!!')
            print()
            app = TextToSpeech(subscription_key)
            app.get_token()


        print()


