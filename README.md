# Audio-Visual-Deep-Multimodal-Networks


### Data

Image Data: Download the ImageNet data (http://www.image-net.org/). </br>
Audio Data: Generate spoken data for all single word ImageNet classes (576). In this paper we use the Google, IBM and Microsoft TTS systems to generate the audio data. Helper scripts are provided in the Preprocessing directory.
```cd Preprocessing```
``` --google_tts.py ```
``` --ibm_tts.py ```
``` --microsoft_tts.py ```

### Pre-processing

We extract 80 dimensional features using the LRE baseline. The features extracted are stored in the form of a dictionary and split into train, validation and test sets. The data along with their corresponding csv files are stored in the Data directory.
To preprocess the data, run:
```bash preprocessing.sh```


### Training


### Testing

### References
