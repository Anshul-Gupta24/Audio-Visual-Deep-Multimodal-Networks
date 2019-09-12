for filename in ../Data/synth_eng_txt/*.txt; do
	label=${filename:69:-4}
	echo $label

	sed 's|audioContent| |' < $filename > tmp-output.txt && \
	tr -d '\n ":{}' < tmp-output.txt > tmp-output-2.txt && \
	base64 tmp-output-2.txt --decode > ../Data/google_synth_eng/synthesize-audio_$label.wav && \
	rm tmp-output*.txt

done
