'''
Script to generate Microsoft TTS data
'''
import os, requests, time
import numpy as np
import pandas as pd
import time
import pandas as pd
from xml.etree import ElementTree


class TextToSpeech(object):
    def __init__(self, subscription_key):
        self.subscription_key = subscription_key
        self.timestr = time.strftime("%Y%m%d-%H%M")
        self.access_token = None


    def get_token(self):
        fetch_token_url = "https://westus.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key
        }
        response = requests.post(fetch_token_url, headers=headers)
        self.access_token = str(response.text)


    def save_audio(self, language, speaker, text, label, number):
        base_url = 'https://westus.tts.speech.microsoft.com/'
        path = 'cognitiveservices/v1'
        constructed_url = base_url + path
        headers = {
            'Authorization': 'Bearer ' + self.access_token,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
            'User-Agent': 'YOUR_RESOURCE_NAME'
        }
        xml_body = ElementTree.Element('speak', version='1.0')
        xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', language)
        voice = ElementTree.SubElement(xml_body, 'voice')
        voice.set('{http://www.w3.org/XML/1998/namespace}lang', language)
        voice.set('name', 'Microsoft Server Speech Text to Speech Voice ('+language+', '+speaker+')')
        voice.text = label
        body = ElementTree.tostring(xml_body)

        response = requests.post(constructed_url, headers=headers, data=body)
        if response.status_code == 200:
            with open('../Data/microsoft_synth_eng/synthesize-audio_' + label + '_' + str(number) + '.wav', 'wb') as audio:
                audio.write(response.content)
                print("\nStatus code: " + str(response.status_code) + "\nYour TTS is ready for playback.\n")
        else:
            print("\nStatus code: " + str(response.status_code) + "\nSomething went wrong. Check your subscription key and headers.\n")



subscription_key = "<enter_your_subscription_key>"
app = TextToSpeech(subscription_key)
app.get_token()


labels = open('../classes.txt','r').read().split('\n')
labels = labels[:-1]


# English voices
accents = ['en-US', 'en-IN','en-AU', 'en-CA', 'en-GB', 'en-IE']
voices = {}
voices['en-US'] = ['ZiraRUS', 'Jessa24KRUS', 'BenjaminRUS', 'Guy24KRUS']
voices['en-IN'] = ['Heera, Apollo', 'PriyaRUS', 'Ravi, Apollo']
voices['en-AU'] = ['Catherine', 'HayleyRUS']
voices['en-CA'] = ['Linda', 'HeatherRUS']
voices['en-GB'] = ['Susan, Apollo', 'HazelRUS', 'George, Apollo']
voices['en-IE'] = ['Sean']

# Japanese voices
# accents = ['ja-JP']
# voices = {}
# voices['ja-JP'] = ['Ayumi, Apollo', 'Ichiro, Apollo', 'HarukaRUS']


done = os.listdir('../Data/microsoft_synth_eng')
count = 0
for i in range(len(eng)):
    if 'synthesize-audio_'+eng[i]+'_0.wav' not in done:
        c = 0
        print(eng[i])
        for a in accents:
            for v in voices[a]:      
                print(v)      
                print(c)
                app.save_audio(a, v, jap[i], eng[i], c)
                c += 1
                count +=1
                time.sleep(3)

        if(count%195==0):
            print()
            print('new token!!!')
            print()
            app = TextToSpeech(subscription_key)
            app.get_token()
