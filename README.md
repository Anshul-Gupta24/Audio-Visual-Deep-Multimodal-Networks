# Audio-Visual-Deep-Multimodal-Networks
#### Code for the paper 'Second Language Transfer Learning in Humans and Machines using Image Supervision' accepted at ASRU 2019.
In this paper we train a proxy based deep multimodal neural network to learn spoken words from a new language, and compare its performance with humans. Training happens in two phases. In the first phase we jointly train our image subnetwork and proxy matrix. The proxies learned represent each image class and are fixed. In the second phase we train the audio subnetwork against the fixed proxies.</br> This repository contains code for training the deep multimodal network.

### Data

Image Data: Download the ImageNet data (http://www.image-net.org/). For the 79 classes not present in ImageNet, download images from Google and Flickr.  </br>
Audio Data: Generate English (classes.txt) and Japanese (new_classes.txt) spoken data. In this paper we use the Google, IBM and Microsoft TTS systems to generate the audio data. </br></br>
Helper scripts are provided in the Preprocessing directory.</br>
```
cd Preprocessing
 --english-japanese.csv
 --google_tts.py 
 --ibm_tts.py 
 --microsoft_tts.py
 
 --download_flickr_image.py
 --download_google_image.py
 ```

### Pre-processing

For the audio files, we extract 80 dimensional features using the LRE baseline. For the image files, we extract 2048 dimensional features using the Xception network. The features extracted are stored in the form of a dictionary and split into train, validation and test sets. The data along with their corresponding csv files are stored in the Data directory.</br></br>
Download the LRE baseline weights from [here](https://drive.google.com/open?id=1RTlIayP658dPTRQhCDklK8lwuo81Utap) and store in the Preprocessing/LRE_baseline/models directory. To extract the features use the provided helper scripts.</br>
```
cd Preprocessing
--prepare_images.py
cd Preprocessing/LRE_baseline
--get_features.py
```
And generate the following pickle files:</br>
```
cd Data
--google_synth_features.pkl
--ibm_synth_features.pkl
--microsoft_synth_features.pkl
--google_synth_features_jap.pkl
--ibm_synth_features_jap.pkl
--microsoft_synth_features_jap.pkl

--image_features_imagenet_train.pkl
--image_features_imagenet_val.pkl
--image_features_google.pkl
--image_features_flickr.pkl
```
Then run:</br>
```
cd Preprocessing
bash preprocessing.sh
```

### Training
#### Pretraining
To train the English audio subnetwork, run:</br>
```
python train_audio_subnetwork.py
```
To train the English speech-image deep multimodal network, run: </br>
```
python train_proxy.py 
```
#### Transfer Learning
To train the Japanese audio subnetwork, run:</br>
```
python train_audio_subnetwork_jap.py
```
Extract the 2048 dimensional Japanese audio embeddings using:</br>
```
cd Preprocessing
python get_speech_features_2048D_jap.py
```
To train the Japanese speech-image deep multimodal network, run: </br>
```
python train_proxy_jap.py 
```


### Testing
To get the top K retrieval accuracy for the English speech-image network, run:
```
python test.py
```
To get the top K retrieval accuracy for the Japanese speech-image network, run:
```
python test_jap.py 
```
</br>


### References
Praveen K.\*, Gupta A.\*, Ganapathy S., Soman A. (2019). 'Second Language Transfer Learning in Humans and Machines using Image Supervision'. Accepted at ASRU 2019. (* equal contribution)
