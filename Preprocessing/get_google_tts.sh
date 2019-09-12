curl -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
  -H "Content-Type: application/json; charset=utf-8" \
  --data "{
    'input':{
      'text':'$1'
    },
    'voice':{
      'languageCode':'$2',
      'name':'$3',
    },
    'audioConfig':{
      'audioEncoding':'LINEAR16',
      'sampleRateHertz': 8000
    }
  }" "https://texttospeech.googleapis.com/v1/text:synthesize" > ../Data/google_synth_eng_txt/synthesize-text_$1_$4.txt
