# First install google_images_download
# pip install google_images_download
from google_images_download import google_images_download
from functions.data import get_label_to_remap
import pandas as pd

response = google_images_download.googleimagesdownload() #class instantiation
output_directory = 'Google_images'

def download(keyword, imgdir):
    image_directory = imgdir
    arguments = {'keywords':keyword, 'limit':150,'print_urls':False,'image_directory':image_directory, 'output_directory':output_directory, 'chromedriver':'<your_path>'}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images

labels = open('../new_classes.txt').read().split('\n')[:-1]
l2r = get_label_to_remap()

for keyword in keywords:
    download(l2r[keyword], keyword)
