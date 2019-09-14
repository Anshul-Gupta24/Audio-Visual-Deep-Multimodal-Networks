# Audio-Visual-Deep-Multimodal-Networks
#### Code for the paper 'Second Language Transfer Learning in Humans and Machines using Image Supervision' accepted at ASRU 2019.

### Data

Image Data: Download the ImageNet data (http://www.image-net.org/). For the 79 classes not present in ImageNet, download images from Google and Flickr.  </br>
Audio Data: Generate spoken data for all classes (classes.txt). In this paper we use the Google, IBM and Microsoft TTS systems to generate the audio data. </br></br>
Helper scripts are provided in the Preprocessing directory.</br>
```
cd Preprocessing
 --google_tts.py 
 --ibm_tts.py 
 --microsoft_tts.py
 ```

### Pre-processing

We extract 80 dimensional features using the LRE baseline. The features extracted are stored in the form of a dictionary and split into train, validation and test sets. The data along with their corresponding csv files are stored in the Data directory.</br></br>
Download the LRE baseline system from () and store in the Preprocessing directory. To preprocess the data, run:</br>
```
cd Preprocessing
bash preprocessing.sh
```


### Training


### Testing

### References
