mkdir ../Saved_models
mkdir ../Plots
python LRE_baseline/get_features.py
python get_speech_features_seq.py
python get_speech_features_seq_jap.py
python create_csv.py
python create_csv_jap.py
