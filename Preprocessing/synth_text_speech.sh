for filename in /home/anshul/keras/WordNet/get_jap_fre/synth_eng_txt/*.txt; do
	#filename='/home/anshul/keras/WordNet/textop/synthesize-text_hourglass_0.txt'
	label=${filename:69:-4}
	echo $label

	sed 's|audioContent| |' < $filename > tmp-output.txt && \
	tr -d '\n ":{}' < tmp-output.txt > tmp-output-2.txt && \
	base64 tmp-output-2.txt --decode > ./synth_eng/synthesize-audio_$label.wav && \
	rm tmp-output*.txt

done
