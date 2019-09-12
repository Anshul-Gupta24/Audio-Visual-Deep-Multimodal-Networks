# text="jack-o-lantern"
text_eng=$1
# text_jap=$2
# echo $2
# number="_0"

voice=$2
number=$3




# NOTE: GET query in the form of {url}?{query}, ex. {url}?voice=en-US_AllisonVoice
# For additional queries use &, ex. {url}?voice=en-US_AllisonVoice&accept=XX
# Accept query in the form of accept=audio/wav;rate=8000... replace special characters with % followed by their HEX codes, ex. accept=audio%2Fwav%3Brate=8000

curl -X POST -u "apikey:SbMajydP0uYGPuwRWD88xW1wR9K5aX9VnxEzevX4ZUvz" --header "Content-Type: application/json" --header "Accept: audio/wav" --data '{"text": "'$text_eng'"}' --output /home/data1/anshulg/ibm_synth_eng_exp/$text_eng$number.wav "https://gateway-tok.watsonplatform.net/text-to-speech/api/v1/synthesize?voice=$2&accept=audio%2Fwav%3Brate=8000"
