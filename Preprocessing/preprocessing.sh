mkdir ../Saved_models
mkdir ../Plots
python get_speech_features_seq.py
python get_speech_features_seq_jap.py
python get_image_features_2048D.py
python create_csv.py
python create_csv_jap.py
