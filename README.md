# Audio-Visual-Deep-Multimodal-Networks
#### Code for the paper 'Second Language Transfer Learning in Humans and Machines using Image Supervision' accepted at ASRU 2019.

### Data

Image Data: Download the ImageNet data (http://www.image-net.org/). For the 79 classes not present in ImageNet, download images from Google and Flickr.  </br>
Audio Data: Generate English (classes.txt) and Japanese (new_classes.txt) spoken data. In this paper we use the Google, IBM and Microsoft TTS systems to generate the audio data. </br></br>
Helper scripts are provided in the Preprocessing directory.</br>
```
cd Preprocessing
 --google_tts.py 
 --ibm_tts.py 
 --microsoft_tts.py
 ```

### Pre-processing

We extract 80 dimensional features using the LRE baseline. The features extracted are stored in the form of a dictionary and split into train, validation and test sets. The data along with their corresponding csv files are stored in the Data directory.</br></br>
Download the LRE baseline weights from [here](https://drive.google.com/open?id=1RTlIayP658dPTRQhCDklK8lwuo81Utap) and store in the Preprocessing/LRE_baseline/models directory. To preprocess the data, run:</br>
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
Extract the 2048 dimensional audio and image embeddings using:</br>
```
cd Preprocessing
python get_speech_features_2048D.py
python get_image_features_2048D.py
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

### References
Praveen K.\*, Gupta A.\*, Ganapathy S., Soman A. (2019). 'Second Language Transfer Learning in Humans and Machines using Image Supervision'. Accepted at ASRU 2019.
