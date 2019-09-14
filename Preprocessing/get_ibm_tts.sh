text_eng=$1
voice=$2
number=$3

curl -X POST -u "apikey:<enter_your_api_key>" --header "Content-Type: application/json" --header "Accept: audio/wav" --data '{"text": "'$text_eng'"}' --output ../Data/ibm_synth_eng/synthesize-audio_$text_eng$number.wav "https://gateway-tok.watsonplatform.net/text-to-speech/api/v1/synthesize?voice=$2&accept=audio%2Fwav%3Brate=8000"
