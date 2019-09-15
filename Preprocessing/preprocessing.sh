mkdir Saved_models
mkdir Plots
python LRE_baseline/get_features.py
python get_speech_features_seq.py
python create_csv.py
